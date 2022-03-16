/* ******************************************************************************************************************************************
 * 3D Point Cloud Mapping Algorithm (2022)																									*
 * @author https://github.com/vyanev1																										*
 * ******************************************************************************************************************************************/

#include <thread>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

// ------------------------------------------------------------ Global Variables ------------------------------------------------------------

typedef pcl::PointXYZRGBA PointT;

pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr cloud_color_filtered(new pcl::PointCloud<PointT>);
pcl::PassThrough<PointT> pass;

static constexpr float SPHERE_RADIUS = 0.03f;

float low_limit = 0.7f;
float high_limit = 1;
bool changed_limits = false;


// ------------------------------------------------------------ Viewer Callbacks ------------------------------------------------------------

void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
	std::cout << "i only run once" << std::endl;
	viewer.initCameraParameters();
}

void viewerPsycho(pcl::visualization::PCLVisualizer& viewer)
{
	std::stringstream ss;
	ss << "Low limit: " << low_limit << "; High limit: " << high_limit;
	viewer.removeShape("text", 0);
	viewer.addText(ss.str(), 200, 300, "text", 0);
	if (changed_limits) {
		pass.setFilterLimits(low_limit, high_limit);
		pass.filter(*cloud);
		viewer.updatePointCloud(cloud);
		changed_limits = false;
	}
}


// ---------------------------------------------------------------- Filters -----------------------------------------------------------------

void applyPassThroughFilter(const pcl::PointCloud<PointT>::Ptr& in_cloud)
{
	cout << "[INFO] Applying PassThrough filter..." << endl;
	pass.setInputCloud(in_cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(low_limit, high_limit);
	pass.setFilterLimitsNegative(false);
	pass.filter(*in_cloud);
}

void applyRGBColorFilter(const pcl::PointCloud<PointT>::Ptr& cloud, const pcl::PointCloud<PointT>::Ptr& cloud_output)
{
	cout << "[INFO] Applying RGB filter for calibration object" << endl;
	const pcl::ConditionAnd<PointT>::Ptr color_condition(new pcl::ConditionAnd<PointT>());
	color_condition->addComparison(boost::make_shared<pcl::PackedRGBComparison<PointT> const>("r", pcl::ComparisonOps::LT, 130));
	color_condition->addComparison(boost::make_shared<pcl::PackedRGBComparison<PointT> const>("g", pcl::ComparisonOps::LT, 130));
	color_condition->addComparison(boost::make_shared<pcl::PackedRGBComparison<PointT> const>("b", pcl::ComparisonOps::GT, 90));

	// Build the filter
	pcl::ConditionalRemoval<PointT> color_filter;
	color_filter.setInputCloud(cloud);
	color_filter.setCondition(color_condition);
	color_filter.filter(*cloud_output);
}


// ----------------------------------------------------------- Normal Estimation ------------------------------------------------------------

void computeNormals(const pcl::PointCloud<PointT>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
	cout << "[INFO] Computing normals..." << endl;
	const pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.01);
	ne.setInputCloud(cloud);
	ne.useSensorOriginAsViewPoint();
	ne.compute(*normals);
}


// ------------------------------------------------------------ Helper Methods --------------------------------------------------------------

void concatenateFields(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud1_in, const pcl::PointCloud<pcl::Normal>& cloud2_in, pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud_out)
{
	if (cloud1_in.points.size() != cloud2_in.points.size())
	{
		PCL_ERROR("[pcl::concatenateFields] The number of points in the two input datasets differs!\n");
		return;
	}

	// Resize the output dataset
	cloud_out.points.resize(cloud1_in.points.size());
	cloud_out.header = cloud1_in.header;
	cloud_out.width = cloud1_in.width;
	cloud_out.height = cloud1_in.height;
	if (!cloud1_in.is_dense || !cloud2_in.is_dense)
		cloud_out.is_dense = false;
	else
		cloud_out.is_dense = true;

	// Iterate over each point
	for (size_t i = 0; i < cloud_out.points.size(); ++i)
	{
		const pcl::PointXYZRGBA& x1 = cloud1_in[i];
		const pcl::Normal& x2 = cloud2_in[i];
		pcl::PointXYZRGBNormal& y = cloud_out[i];
		memcpy(y.data, x1.data, sizeof(x1.data));			// coordinates
		memcpy(y.data_n, x2.data_n, sizeof(x2.data_n));	// normal
		y.rgba = x1.rgba;										// color
		y.curvature = x2.curvature;								// curvature
	}
}

pcl::PointXYZRGBNormal getNearestPoint(pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud)
{
	cout << "[INFO] Finding nearest point..." << endl;

	float min_distance = FLT_MAX;
	pcl::PointXYZRGBNormal min_distance_point;
	for (const auto& point : cloud.points)
	{
		cout << "  " << min_distance_point.x << " " << min_distance_point.y << " " << min_distance_point.z << endl;
		if (point.z > low_limit && point.z < min_distance)
		{
			min_distance = point.z;
			min_distance_point = point;
		}
	}
	cout << "coords:  " << min_distance_point.x << " " << min_distance_point.y << " " << min_distance_point.z << endl;
	cout << "normals: " << min_distance_point.normal_x << " " << min_distance_point.normal_y << " " << min_distance_point.normal_z << endl;

	return min_distance_point;
}


// ----------------------------------------------------------- Sphere Generation ------------------------------------------------------------

void generateSphere(const pcl::PointXYZRGBNormal input_point, const float radius, const char color, const pcl::PointCloud<PointT>::Ptr& sphere_cloud)
{
	cout << "[INFO] Drawing sphere..." << endl;
	int index = 0;
	for (size_t m = 0; m < sphere_cloud->height; m++)
	{
		for (size_t n = 0; n < sphere_cloud->width - 1; n++)
		{
			const double x = std::sin(M_PI * static_cast<double>(m) / sphere_cloud->height) * std::cos(2 * M_PI * static_cast<double>(n) / sphere_cloud->width);
			const double y = std::sin(M_PI * static_cast<double>(m) / sphere_cloud->height) * std::sin(2 * M_PI * static_cast<double>(n) / sphere_cloud->width);
			const double z = std::cos(M_PI * static_cast<double>(m) / sphere_cloud->height);

			sphere_cloud->points[index].x = (input_point.x - radius * input_point.normal_x) + (static_cast<float>(x) * radius);
			sphere_cloud->points[index].y = (input_point.y - radius * input_point.normal_y) + (static_cast<float>(y) * radius);
			sphere_cloud->points[index].z = (input_point.z - radius * input_point.normal_z) + (static_cast<float>(z) * radius);

			switch (color)
			{
				case 'R': sphere_cloud->points[index].r = 255;
				case 'G': sphere_cloud->points[index].g = 255;
				case 'B': sphere_cloud->points[index].b = 255;
				default:;
			}

			index++;
		}
	}
}


int main()
{
	const std::string colors = "RGB";
	const pcl::PointCloud<PointT>::Ptr curr_cloud(new pcl::PointCloud<PointT>);
	for (int camera_index = 1; camera_index < 2; camera_index++)
	{
		for (int color_index = 0; color_index < colors.size(); color_index++)
		{
			// E.g. "camera_1_R.pcd", "camera_1_G.pcd", "camera_1_B.pcd"
			std::string filename = "camera_" + std::to_string(camera_index) + "_" + colors[color_index] + ".pcd";

			cout << "[INFO] Fetching " << filename << "..." << endl;
			pcl::io::loadPCDFile(filename, *curr_cloud);

			cout << "[INFO] Removing NaN points..." << endl;
			std::vector<int> indices;
			removeNaNFromPointCloud(*curr_cloud, *curr_cloud, indices);

			cout << "[INFO] Inverting X & Y coordinates..." << endl;
			for (auto& point : curr_cloud->points)
			{
				point.x *= -1;
				point.y *= -1;
			}

			cout << "[INFO] Applying PassThrough filter..." << endl;
			applyPassThroughFilter(curr_cloud);

			applyRGBColorFilter(curr_cloud, cloud_color_filtered);

			const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
			const pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
			computeNormals(cloud_color_filtered, normals);
			concatenateFields(*cloud_color_filtered, *normals, *cloud_with_normals);

			const pcl::PointCloud<PointT>::Ptr sphere_cloud(new pcl::PointCloud<PointT>(30, 30));
			const pcl::PointXYZRGBNormal nearest_point = getNearestPoint(*cloud_with_normals);
			generateSphere(nearest_point, SPHERE_RADIUS, colors[color_index], sphere_cloud);
			*cloud_color_filtered += *sphere_cloud;

			cout << "[INFO] Merging " << filename << " to main PointCloud object" << endl;
			*cloud += *cloud_color_filtered;
		}
	}

	cout << "[INFO] Opening CloudViewer..." << endl;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(cloud);
	viewer.runOnVisualizationThreadOnce(viewerOneOff); // This will only get called once
	viewer.runOnVisualizationThread(viewerPsycho);     // This will get called once per visualization iteration

	while (!viewer.wasStopped())
	{
		if (getchar() != 0)
		{
			low_limit += 0.1f;
			high_limit += 0.1f;
			changed_limits = true;
		}
	}

	return 0;
}
