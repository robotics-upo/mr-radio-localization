#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <chrono>
#include <pcl/visualization/pcl_visualizer.h>
#include <X11/Xlib.h>

// Include the service header (adjust the package name accordingly)
#include "uwb_localization/srv/update_point_clouds.hpp"  
using UpdatePointClouds = uwb_localization::srv::UpdatePointClouds;

class PCLVisualizerNode : public rclcpp::Node {
public:
  PCLVisualizerNode() : Node("pcl_visualizer_node") {
    // Create the PCL visualizer.
    viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("ICP Viewer");
    viewer_->setBackgroundColor(0, 0, 0);
    viewer_->initCameraParameters();

    pcl_viewer_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100), std::bind(&PCLVisualizerNode::pcl_viewer_cb_, this));

    // Create a service to update the visualizer with new point clouds.
    update_service_ = this->create_service<UpdatePointClouds>(
      "eliko_optimization_node/pcl_visualizer_service",
      std::bind(&PCLVisualizerNode::updateVisualizerCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "PCL Visualizer Node initialized.");
  } 

private:
    // Timer callback to spin the viewer.
    void pcl_viewer_cb_(){
        if (viewer_ && !viewer_->wasStopped()) {
        viewer_->spinOnce(10);
        }
    }

    // Modified updateVisualizer function to include the aligned cloud.
    void updateVisualizer(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &source_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &target_cloud, 
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &aligned_cloud,
        const std::string &source_cloud_id, 
        const std::string &target_cloud_id,
        const std::string &aligned_cloud_id) const
    {
        if (!viewer_ || !source_cloud || !target_cloud || !aligned_cloud) return;

        // Set up color handlers for each cloud.
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> 
        source_color(source_cloud, 255, 0, 0);  // Red for source.
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> 
        target_color(target_cloud, 0, 255, 0);    // Green for target.
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> 
        aligned_color(aligned_cloud, 0, 0, 255);    // Blue for aligned.

        // // Update source cloud.
        // if (!viewer_->contains(source_cloud_id)) {
        // viewer_->addPointCloud<pcl::PointXYZ>(source_cloud, source_color, source_cloud_id);
        // viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, source_cloud_id);
        // } else {
        // viewer_->updatePointCloud<pcl::PointXYZ>(source_cloud, source_color, source_cloud_id);
        // }

        // // Update target cloud.
        // if (!viewer_->contains(target_cloud_id)) {
        // viewer_->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, target_cloud_id);
        // viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, target_cloud_id);
        // } else {
        // viewer_->updatePointCloud<pcl::PointXYZ>(target_cloud, target_color, target_cloud_id);
        // }
        
        // Update aligned cloud.
        if (!viewer_->contains(aligned_cloud_id)) {
        viewer_->addPointCloud<pcl::PointXYZ>(aligned_cloud, aligned_color, aligned_cloud_id);
        viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, aligned_cloud_id);
        } else {
        viewer_->updatePointCloud<pcl::PointXYZ>(aligned_cloud, aligned_color, aligned_cloud_id);
        }
    }

    // The service callback that receives new point clouds and updates the visualizer.
    void updateVisualizerCallback(
        const std::shared_ptr<UpdatePointClouds::Request> request,
        std::shared_ptr<UpdatePointClouds::Response> response)
    {
        // Convert ROS PointCloud2 messages to PCL point clouds.
        pcl::PointCloud<pcl::PointXYZ>::Ptr new_source(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr new_target(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr new_aligned(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::fromROSMsg(request->source_cloud, *new_source);
        pcl::fromROSMsg(request->target_cloud, *new_target);
        pcl::fromROSMsg(request->aligned_cloud, *new_aligned);

        // Update the internal point cloud pointers.
        source_cloud_ = new_source;
        target_cloud_ = new_target;
        aligned_cloud_ = new_aligned;

        // Update the visualizer display.
        updateVisualizer(source_cloud_, target_cloud_, aligned_cloud_, "source_cloud", "target_cloud", "aligned_cloud");

        response->success = true;
        response->message = "Point clouds updated successfully";

        RCLCPP_INFO(this->get_logger(), "Visualizer updated via service call.");
    }

    void saveScreenshot(const std::string &filename) {
        if (viewer_) {
        viewer_->saveScreenshot(filename);
        }
    }
    
    rclcpp::TimerBase::SharedPtr pcl_viewer_timer_;

    // Service server for updating point clouds in the visualizer.
    rclcpp::Service<UpdatePointClouds>::SharedPtr update_service_;

    // Internal storage for the point clouds.
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_{new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_{new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_{new pcl::PointCloud<pcl::PointXYZ>};

    // The PCL visualizer pointer.
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
    };

    int main(int argc, char** argv) {
    if (!XInitThreads()) {
        fprintf(stderr, "Failed to initialize Xlib multithreading.\n");
        return -1;
    }

    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCLVisualizerNode>();
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
