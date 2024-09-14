/* Based on https://cgal.geometryfactory.com/CGAL/doc/master/
Optimal_bounding_box/index.html#OBBBasicExample.
*/
#define CGAL_EIGEN3_ENABLED
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/optimal_bounding_box.h>

#include <fstream>
#include <iostream>
namespace PMP = CGAL::Polygon_mesh_processing;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;
int main(int argc, char **argv) {
  const std::string filename = CGAL::data_file_path(argv[1]);
  Point_set ps;
  bool file_loaded = CGAL::IO::read_PLY(filename, ps);
  if (!file_loaded || ps.is_empty()) {
    std::cerr << "Invalid input file '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }
  // Compute the extreme points of the mesh, and then a tightly fitted oriented
  // bounding box.
  std::array<Point, 8> obb_points;
  std::vector<Point> points;
  for (const auto &v : ps) {
    points.push_back(ps.point(v));
  }
  CGAL::oriented_bounding_box(points, obb_points,
                              CGAL::parameters::use_convex_hull(true));
  std::string new_filename =
      filename.substr(0, filename.find(".ply")) + "_bbox.txt";
  std::ofstream out(new_filename);
  for (const auto &obb_point : obb_points) {
    out << obb_point << std::endl;
  }
  return EXIT_SUCCESS;
}