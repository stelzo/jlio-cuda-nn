// #include <algorithm>
#include <cmath>
#include <list>
// #include <vector>
#include <cstddef>
#include <iostream>
#include <limits>
#include <cuda_runtime.h>

#include "../point.cuh"
#include "../math/math.cuh"
#include <voxel/hilbert.cuh>

#include <thrust/device_reference.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/pair.h>
#include <cuco/dynamic_map.cuh>

#include <unordered_map>

// There are 2 levels of voxels seen as layers.

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <thrust/device_ptr.h>

#include <limits>

void test()
{
    // Create a device_vector of integers
    thrust::device_vector<int> d_vector(10, 1);

    // Create a device_ptr to the data inside the device_vector
    thrust::device_ptr<int> d_ptr = d_vector.data();
}

// General helpers
struct less_vec
{
    inline bool operator()(const rmagine::Vector3i &v1, const rmagine::Vector3i &v2) const;
};

struct hash_vec
{
    inline size_t operator()(const rmagine::Vector3i &v) const;
};

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>

struct CentroidPoint
{
    float sum_x_ = 0;
    float sum_y_ = 0;
    float sum_z_ = 0;
    int count_ = 0;

    void add(const jlio::PointXYZINormal &pt);

    jlio::PointXYZINormal get() const;
};
// END General helpers

// -- PHC --
class IVoxNodePhc
{
public:
    struct DistPoint;
    struct PhcCube;

    IVoxNodePhc() = default;
    IVoxNodePhc(const jlio::PointXYZINormal &center, const float &side_length, const int &phc_order = 6);

    void InsertPoint(const jlio::PointXYZINormal &pt);

    void ErasePoint(const jlio::PointXYZINormal &pt, const double erase_distance_th_);

    inline bool Empty() const;

    inline std::size_t Size() const;

    jlio::PointXYZINormal GetPoint(const std::size_t idx) const;

    bool NNPoint(const jlio::PointXYZINormal &cur_pt, DistPoint &dist_point) const;

    int KNNPointByCondition(DistPoint *dis_points, const size_t dis_points_size, const jlio::PointXYZINormal &cur_pt, const int &K = 5,
                            const double &max_range = 5.0);

private:
    uint32_t CalculatePhcIndex(const jlio::PointXYZINormal &pt) const;

private:
    thrust::device_vector<PhcCube> phc_cubes_;

    jlio::PointXYZINormal center_;
    float side_length_ = 0;
    int phc_order_ = 6;
    float phc_side_length_ = 0;
    float phc_side_length_inv_ = 0;
    rmagine::Vector3<float> min_cube_;
};

struct IVoxNodePhc::DistPoint
{
    float dist = 0;
    const IVoxNodePhc *node = nullptr;
    int idx = 0;

    DistPoint() {}
    DistPoint(const float d, const IVoxNodePhc *n, const int i) : dist(d), node(n), idx(i) {}

    jlio::PointXYZINormal Get() { return node->GetPoint(idx); }

    inline bool operator()(const DistPoint &p1, const DistPoint &p2) { return p1.dist < p2.dist; }

    inline bool operator<(const DistPoint &rhs) { return dist < rhs.dist; }
};

struct IVoxNodePhc::PhcCube
{
    uint32_t idx = 0;
    CentroidPoint mean;

    PhcCube(uint32_t index, const jlio::PointXYZINormal &pt);

    void AddPoint(const jlio::PointXYZINormal &pt);

    jlio::PointXYZINormal GetPoint() const;
};
// -- END PHC --

// -- Linear --
class IVoxNode
{
public:
    struct DistPoint;

    IVoxNode() = default;
    IVoxNode(const jlio::PointXYZINormal &center, const float &side_length) {} /// same with phc

    void InsertPoint(const jlio::PointXYZINormal &pt);

    inline bool Empty() const;

    inline std::size_t Size() const;

    inline jlio::PointXYZINormal GetPoint(const std::size_t idx) const;

    int KNNPointByCondition(DistPoint *dis_points, const size_t dis_points_size, const jlio::PointXYZINormal &point, const int &K,
                            const double &max_range);

private:
    jlio::PointXYZINormal points_[100]; // TODO do not know size, how many are inside a voxel? maybe ringbuffer is better
};

struct IVoxNode::DistPoint
{
    double dist = 0;
    IVoxNode *node = nullptr;
    int idx = 0;

    DistPoint() = default;
    DistPoint(const double d, IVoxNode *n, const int i) : dist(d), node(n), idx(i) {}

    jlio::PointXYZINormal Get() { return node->GetPoint(idx); }

    inline bool operator()(const DistPoint &p1, const DistPoint &p2) { return p1.dist < p2.dist; }

    inline bool operator<(const DistPoint &rhs) { return dist < rhs.dist; }
};
// -- END Linear --

// Node traits
enum class IVoxNodeType
{
    DEFAULT, // linear ivox
    PHC,     // phc ivox
};

template <IVoxNodeType node_type, typename PointT, int dim>
struct IVoxNodeTypeTraits
{
};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim>
{
    using NodeType = IVoxNode; // ignore PointT and dim, is always the same
};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::PHC, PointT, dim>
{
    using NodeType = IVoxNodePhc;
};
// END Node traits

// Main datastructure wrapper
class IVox
{
public:
    using KeyType = rmagine::Vector3i;
    using PtType = rmagine::Vector3f;
    using DistPoint = typename IVoxNodePhc::DistPoint;

    using MapValueT = thrust::pair<KeyType, IVoxNodePhc> *;
    using MapKeyT = size_t;

    enum class NearbyType
    {
        CENTER, // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    struct Options
    {
        float resolution_ = 0.2;                       // ivox resolution
        float inv_resolution_ = 10.0;                  // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6; // nearby range
        std::size_t capacity_ = 1000000;               // capacity
    };

    /**
     * constructor
     * @param options  ivox options
     */
    explicit IVox(Options options) : options_(options)
    {
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();

        constexpr std::size_t N = 50'000;

        // Empty slots are represented by reserved "sentinel" values. These values should be selected such
        // that they never occur in your input data.
        MapKeyT constexpr empty_key_sentinel = 0;
        MapValueT constexpr empty_value_sentinel = 0;

        // Number of key/value pairs to be inserted
        std::size_t constexpr num_keys = 50'000;

        // Compute capacity based on a 50% load factor
        auto constexpr load_factor = 0.5;
        std::size_t const capacity = std::ceil(num_keys / load_factor);

        int empty_key_sentinel = -1;
        int empty_value_sentinel = -1;

        // Constructs a map with "capacity" slots using -1 and -1 as the empty key/value sentinels.
        cuco::dynamic_map<MapKeyT, MapValueT> m{100'000, cuco::empty_key<MapKeyT>{empty_key_sentinel}, cuco::empty_value<MapValueT>{empty_value_sentinel}};
        // TODO move into memory, keep ptr
    }

    /**
     * add points
     * @param points_to_add
     */
    JLIO_FUNCTION
    void AddPoints(const jlio::PointXYZINormal *points_to_add, const size_t size);

    /// get nn
    JLIO_FUNCTION
    bool GetClosestPoint(const jlio::PointXYZINormal &pt, jlio::PointXYZINormal &closest_pt);

    /// get nn with condition
    JLIO_FUNCTION
    bool GetClosestPoint(const jlio::PointXYZINormal &pt, DistPoint *candidates, size_t *candidates_size, jlio::PointXYZINormal *closest_pt, size_t *closest_pt_size, size_t &found_closest_n, double max_range = 5.0);

    /// get number of points
    size_t NumPoints() const;

    /// get number of valid grids
    size_t NumValidGrids() const;

    /// get statistics of the points
    std::vector<float> StatGridPoints() const;

private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    /// position to grid
    KeyType Pos2Grid(const PtType &pt) const;

    Options options_;

    cuco::dynamic_map<MapKeyT, MapValueT> *grids_map_;          // voxel hash map. initial size must be larger (cloud_n / voxel_size) size minimum
    std::list<thrust::pair<KeyType, IVoxNodePhc>> grids_cache_; // voxel cache
    KeyType nearby_grids_[26];                                  // nearbys
    size_t nearby_grids_size_ = 0;                              // nearbys size
};
