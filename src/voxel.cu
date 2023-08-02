#define USE_CUDA 1

#include "voxel/voxel.cuh"

#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

constexpr int K_NEAREST = 5;

void CentroidPoint::add(const jlio::PointXYZINormal& pt)
{
    sum_x_ += pt.x;
    sum_y_ += pt.y;
    sum_z_ += pt.z;
    count_++;
}

jlio::PointXYZINormal CentroidPoint:: get() const {
    jlio::PointXYZINormal pt;
    pt.x = sum_x_ / count_;
    pt.y = sum_y_ / count_;
    pt.z = sum_z_ / count_;
    return std::move(pt);
}

IVoxNodePhc::PhcCube::PhcCube(uint32_t index, const jlio::PointXYZINormal& pt) { mean.add(pt); }

void IVoxNodePhc::PhcCube::AddPoint(const jlio::PointXYZINormal& pt) { mean.add(pt); }

jlio::PointXYZINormal IVoxNodePhc::PhcCube::GetPoint() const { return std::move(mean.get()); }

IVoxNodePhc::IVoxNodePhc(const jlio::PointXYZINormal& center, const float& side_length, const int& phc_order)
    : center_(center), side_length_(side_length), phc_order_(phc_order) {
    assert(phc_order <= 8);
    phc_side_length_ = side_length_ / (std::pow(2, phc_order_));
    phc_side_length_inv_ = (std::pow(2, phc_order_)) / side_length_;
    float half_side_length = side_length_ / 2.0;
    min_cube_ = rmagine::Vector3f(center.x - half_side_length, center.y - half_side_length, center.z - half_side_length);
    phc_cubes_.reserve(64);
}

struct insert_point_lb_functor
{
    __host__ __device__
    bool operator()(const IVoxNodePhc::PhcCube& a, const IVoxNodePhc::PhcCube& b) const
    {
        return a.idx < b.idx;
    }
};

void IVoxNodePhc::InsertPoint(const jlio::PointXYZINormal& pt) {
    uint32_t idx = CalculatePhcIndex(pt);

    PhcCube cube{idx, pt};
    auto it = thrust::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube, insert_point_lb_functor());

    if (it == phc_cubes_.end()) {
        phc_cubes_.push_back(cube);
    } else {
        auto found_it = thrust::raw_pointer_cast(it);
        if (found_it->idx == idx) {
            found_it->AddPoint(pt);
        } else {
            if (found_it->idx == idx) {
                found_it->AddPoint(pt);
            } else {
                phc_cubes_.insert(it, cube);
            }
        }
    }
}

void IVoxNodePhc::ErasePoint(const jlio::PointXYZINormal& pt, const double erase_distance_th_) {
    uint32_t idx = CalculatePhcIndex(pt);

    PhcCube cube{idx, pt};
    auto it = thrust::lower_bound(phc_cubes_.begin(), phc_cubes_.end(), cube, insert_point_lb_functor());

    auto found_it = thrust::raw_pointer_cast(it);
    if (it != phc_cubes_.end() && found_it->idx == idx) {
        phc_cubes_.erase(it);
    }
}

bool IVoxNodePhc::Empty() const {
    return phc_cubes_.empty();
}

std::size_t IVoxNodePhc::Size() const {
    return phc_cubes_.size();
}

jlio::PointXYZINormal IVoxNodePhc::GetPoint(const std::size_t idx) const {
    IVoxNodePhc::PhcCube tmp = phc_cubes_[idx];
    return tmp.GetPoint();
}

bool IVoxNodePhc::NNPoint(const jlio::PointXYZINormal& cur_pt, DistPoint& dist_point) const {
    if (phc_cubes_.empty()) {
        return false;
    }
    uint32_t cur_idx = CalculatePhcIndex(cur_pt);
    PhcCube cube{cur_idx, cur_pt};
    auto it = thrust::lower_bound(phc_cubes_.begin(), phc_cubes_.device_end(), cube, insert_point_lb_functor());

    if (it == phc_cubes_.device_end()) {
        it--;
        auto found_it = thrust::raw_pointer_cast(it);
        dist_point = DistPoint(cur_pt.distance(found_it->GetPoint()), this, it - phc_cubes_.device_begin());
    } else if (it == phc_cubes_.device_begin()) {
        auto found_it = thrust::raw_pointer_cast(it);
        dist_point = DistPoint(cur_pt.distance(found_it->GetPoint()), this, it - phc_cubes_.device_begin());
    } else {
        auto last_it = it;
        last_it--;
        auto found_it = thrust::raw_pointer_cast(it);
        auto found_last_it = thrust::raw_pointer_cast(last_it);
        float d1 = cur_pt.distance(found_it->GetPoint());
        float d2 = cur_pt.distance(found_last_it->GetPoint());
        dist_point = DistPoint(std::min(d1, d2), this, it - phc_cubes_.device_begin());
    }

    return true;
}

int IVoxNodePhc::KNNPointByCondition(stdgpu::vector<DistPoint>& dis_points, const jlio::PointXYZINormal& cur_pt,
                                                  const int& K, const double& max_range) {
    uint32_t cur_idx = CalculatePhcIndex(cur_pt);
    PhcCube cube{cur_idx, cur_pt};
    auto it = thrust::lower_bound(phc_cubes_.device_begin(), phc_cubes_.device_end(), cube, insert_point_lb_functor());

    const int max_search_cube_side_length = std::pow(2, std::ceil(std::log2(max_range * phc_side_length_inv_)));
    const int max_search_idx_th = 8 * max_search_cube_side_length * max_search_cube_side_length * max_search_cube_side_length;

    auto forward_it = it;
    auto backward_it = it - 1;
    if (it != phc_cubes_.device_end()) {
        auto found_it = thrust::raw_pointer_cast(it);
        float d = found_it->GetPoint().distance(cur_pt);
        dis_points.emplace_back(DistPoint(d, this, it - phc_cubes_.device_begin()));
        forward_it++;
    }
    if (backward_it != phc_cubes_.device_begin() - 1) {
        backward_it--;
    }

    auto forward_reach_boundary = [&]() {
        auto found_it = thrust::raw_pointer_cast(forward_it);
        return forward_it == phc_cubes_.device_end() || found_it->idx - cur_idx > max_search_idx_th;
    };
    auto backward_reach_boundary = [&]() {
        auto found_it = thrust::raw_pointer_cast(backward_it);
        return backward_it == phc_cubes_.device_begin() - 1 || cur_idx - found_it->idx > max_search_idx_th;
    };

    while (!forward_reach_boundary() && !backward_reach_boundary()) {
        auto found_forward_it = thrust::raw_pointer_cast(forward_it);
        auto found_backward_it = thrust::raw_pointer_cast(backward_it);
        if (found_forward_it->idx - cur_idx > cur_idx - found_backward_it->idx) {
            float d = found_forward_it->GetPoint().distance(cur_pt);
            dis_points.emplace_back(DistPoint(d, this, forward_it - phc_cubes_.device_begin()));
            forward_it++;
        } else {
            auto found_it = thrust::raw_pointer_cast(backward_it + 1);
            float d = found_it->GetPoint().distance(cur_pt);
            dis_points.emplace_back(DistPoint(d, this, backward_it + 1 - phc_cubes_.device_begin()));
            backward_it--;
        }
        if (dis_points.size() > K) {
            break;
        }
    }

    if (forward_reach_boundary()) {
        while (!backward_reach_boundary() && dis_points.size() < K) {
            auto found_it = thrust::raw_pointer_cast(backward_it + 1);
            float d = found_it->GetPoint().distance(cur_pt);
            dis_points.emplace_back(DistPoint(d, this, backward_it + 1 - phc_cubes_.device_begin()));
            backward_it--;
        }
    }

    if (backward_reach_boundary()) {
        while (!forward_reach_boundary() && dis_points.size() < K) {
            auto found_it = thrust::raw_pointer_cast(forward_it);
            float d = found_it->GetPoint().distance(cur_pt);
            dis_points.emplace_back(DistPoint(d, this, forward_it - phc_cubes_.device_begin()));
            forward_it++;
        }
    }


    return dis_points.size();
}

uint32_t IVoxNodePhc::CalculatePhcIndex(const jlio::PointXYZINormal& pt) const {
    rmagine::Vector3f pt_vec = rmagine::Vector3f(pt.x, pt.y, pt.z);
    rmagine::Vector3f eposf = (pt_vec - min_cube_) * phc_side_length_inv_;
    rmagine::Vector3i eposi = rmagine::Vector3i(static_cast<int>(eposf.x), static_cast<int>(eposf.y), static_cast<int>(eposf.z));

    const int upper_bound = std::pow(2, phc_order_);
    if (eposi.x < 0) {
        eposi.x = 0;
    }
    if (eposi.x > upper_bound) {
        eposi.x = upper_bound - 1;
    }

    if (eposi.y < 0) {
        eposi.y = 0;
    }
    if (eposi.y > upper_bound) {
        eposi.y = upper_bound - 1;
    }

    if (eposi.z < 0) {
        eposi.z = 0;
    }
    if (eposi.z > upper_bound) {
        eposi.z = upper_bound - 1;
    }

    std::array<uint8_t, 3> apos{static_cast<uint8_t>(eposi.x), static_cast<uint8_t>(eposi.y), static_cast<uint8_t>(eposi.z)};
    std::array<uint8_t, 3> tmp = hilbert::v2::PositionToIndex(apos);

    uint32_t idx = (uint32_t(tmp[0]) << 16) + (uint32_t(tmp[1]) << 8) + (uint32_t(tmp[2]));
    return idx;
}

JLIO_FUNCTION
bool IVox::GetClosestPoint(const jlio::PointXYZINormal& pt, jlio::PointXYZINormal& closest_pt) {
    stdgpu::vector<DistPoint> candidates;
    auto key = Pos2Grid(rmagine::Vector3f(pt.x, pt.y, pt.z));
    auto it = nearby_grids_.device_begin();
    // pragma unroll 26
    /*for (size_t i = 0; i < 26, i++) {
        if (i != nearby_grids_size_) {
            auto dkey = key + nearby_grids_[i];
            auto iter = grids_map_.find(dkey);
            if (iter != grids_map_.end()) {
                DistPoint dist_point;
                bool found = iter->second->second.NNPoint(pt, dist_point);
                if (found) {
                    candidates.emplace_back(dist_point);
                }
            }
        }
    }*/
    while(it != nearby_grids_.device_end()) {
        auto raw_delta = thrust::raw_pointer_cast(it);
        auto dkey = key + *raw_delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            DistPoint dist_point;
            bool found = iter->second->second.NNPoint(pt, dist_point);
            if (found) {
                candidates.emplace_back(dist_point);
            }
        }
        it++;
    };

    if (candidates.empty()) {
        return false;
    }

    // min element on device
    auto iter = thrust::min_element(candidates.device_begin(), candidates.device_end());
    auto found_it = thrust::raw_pointer_cast(iter);
    closest_pt = found_it->Get();
    return true;
}

JLIO_FUNCTION
bool IVox::GetClosestPoint(const jlio::PointXYZINormal& pt, stdgpu::vector<DistPoint>& candidates, stdgpu::vector<jlio::PointXYZINormal>& closest_pt, size_t &found_closest_n, double max_range)
{
    auto key = Pos2Grid(rmagine::Vector3f(pt.x, pt.y, pt.z));

    auto it = nearby_grids_.device_begin();
    while (it != nearby_grids_.device_end()) {
        auto raw_it = thrust::raw_pointer_cast(it);
        const KeyType& delta = *raw_it;
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            auto raw_iter = thrust::raw_pointer_cast(iter);
            auto tmp = raw_iter->second->second.KNNPointByCondition(candidates, pt, K_NEAREST, max_range);
        }
    }

    if (candidates.empty()) {
        return false;
    }

    thrust::sort(candidates.device_begin(), candidates.device_end()); // no nth_element on cuda

    found_closest_n = 0;
    auto candidate_it = candidates.device_begin();
    while (candidate_it != candidates.device_end()) {
        auto raw_it = thrust::raw_pointer_cast(candidate_it);
        closest_pt[found_closest_n++] = raw_it->Get();
        candidate_it++;
    }

    return found_closest_n != 0;
}

size_t IVox::NumValidGrids() const {
    return grids_map_.size();
}

void IVox::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, 0);
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, -1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, 1);
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 1, 0);

        nearby_grids_[nearby_grids_size_++] = KeyType(0, -1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 1, 0);

        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, -1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, -1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 0, 1);

        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 0, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 0, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 0, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 1, 1);

        nearby_grids_[nearby_grids_size_++] = KeyType(0, -1, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 1, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, -1, -1);

    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 0, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 1, 0);

        nearby_grids_[nearby_grids_size_++] = KeyType(0, -1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 0, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 1, 0);

        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, -1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, -1, 0);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 0, 1);

        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 0, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 0, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 0, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 1, 1);

        nearby_grids_[nearby_grids_size_++] = KeyType(0, -1, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, 1, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(0, -1, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 1, 1);

        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 1, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, -1, 1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, 1, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, -1, 1);

        nearby_grids_[nearby_grids_size_++] = KeyType(-1, 1, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(1, -1, -1);
        nearby_grids_[nearby_grids_size_++] = KeyType(-1, -1, -1);

    } else {
        printf("Unknown nearby type\n");
    }
}

/*
struct printf_functor
{
  JLIO_INLINE_DEVICE_HOST
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};*/

void IVox::AddPoints(const thrust::device_vector<jlio::PointXYZINormal>& points_to_add)
{

    // thrust hash points to grids

    // thrust make into bins with same keys

    // thrust filter: find each hash in grid, if exists, add all points from bin to voxel [key]->second.AddPoint(point) and filter out. if not exists, let the bin pass through

    // parallel insert

    // add to cache iteratively for now TODO

    // TODO erst so lassen mit grids_cache auf cpu und insert einzeln. muss in unified mem

    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const auto& pt) {
        auto key = Pos2Grid(ToEigen<float, dim>(pt));

        auto iter = grids_map_.find(key);
        if (iter == grids_map_.end()) {
            PointType center;
            center.getVector3fMap() = key.template cast<float>() * options_.resolution_;

            grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
            grids_map_.insert({key, grids_cache_.begin()});

            grids_cache_.front().second.InsertPoint(pt);

            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
            }
        } else {
            iter->second->second.InsertPoint(pt);
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            grids_map_[key] = grids_cache_.begin();
        }
    });
}

rmagine::Vector3i IVox::Pos2Grid(const IVox::PtType& pt) const {
    const int x = static_cast<int>(roundf(pt.x * options_.inv_resolution_));
    const int y = static_cast<int>(roundf(pt.y * options_.inv_resolution_));
    const int z = static_cast<int>(roundf(pt.z * options_.inv_resolution_));
    return rmagine::Vector3i(x, y, z);
}

inline bool less_vec::operator()(const rmagine::Vector3i& v1, const rmagine::Vector3i& v2) const {
    return v1.x < v2.x || ((v1.x == v2.x && v1.y < v2.y) && (v1.x == v2.x && v1.y == v2.y && v1.z < v2.z));
}

inline size_t hash_vec::operator()(const rmagine::Vector3i& v) const {
    return size_t(((v.x) * 73856093) ^ ((v.y) * 471943) ^ ((v.z) * 83492791)) % 10000000;
}

