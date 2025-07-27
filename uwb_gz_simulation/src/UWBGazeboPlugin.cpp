#include "uwb_gz_simulation/UWBGazeboPlugin.hpp"

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/float64_multi_array.hpp>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/components/Pose.hh>

namespace uwb_gz_simulation
{

void UWBGazeboPlugin::Configure(const gz::sim::Entity &entity,
                                const std::shared_ptr<const sdf::Element> &,
                                gz::sim::EntityComponentManager &,
                                gz::sim::EventManager &)
{
  if (!rclcpp::ok())
    rclcpp::init(0, nullptr);

  this->node_ = std::make_shared<rclcpp::Node>("uwb_plugin_node");
  this->publisher_ = this->node_->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/uwb_distances", 10);
}

void UWBGazeboPlugin::PreUpdate(const gz::sim::UpdateInfo &,
                                gz::sim::EntityComponentManager &ecm)
{
  std_msgs::msg::Float64MultiArray msg;
  // ... your logic here ...
  this->publisher_->publish(msg);
  rclcpp::spin_some(this->node_);
}

} // namespace uwb_gz_simulation

GZ_ADD_PLUGIN(uwb_gz_simulation::UWBGazeboPlugin,
              gz::sim::System,
              uwb_gz_simulation::UWBGazeboPlugin::ISystemConfigure,
              uwb_gz_simulation::UWBGazeboPlugin::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(uwb_gz_simulation::UWBGazeboPlugin, "uwb_gz_simulation::UWBGazeboPlugin")
