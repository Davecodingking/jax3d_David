import os
import json
import numpy
import jax
import jax.numpy as np
from PIL import Image
from multiprocessing.pool import ThreadPool
import trimesh


scene_type = "synthetic"
object_name = "MyNeRFData"
scene_dir = "data/custom/" + object_name
mesh_name = "sponza_gt.obj"
mesh_path = os.path.join(scene_dir, mesh_name)
output_path = os.path.join(scene_dir, "mesh_uv_samples.npz")


def matmul(a, b):
  return np.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def normalize(x):
  return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sinusoidal_encoding(position, minimum_frequency_power,
                        maximum_frequency_power, include_identity=False):
  frequency = 2.0**np.arange(minimum_frequency_power, maximum_frequency_power)
  angle = position[..., None, :] * frequency[:, None]
  encoding = np.sin(np.stack([angle, angle + 0.5 * np.pi], axis=-2))
  encoding = encoding.reshape(*position.shape[:-1], -1)
  if include_identity:
    encoding = np.concatenate([position, encoding], axis=-1)
  return encoding


def generate_rays(pixel_coords, pix2cam, cam2world):
  homog = np.ones_like(pixel_coords[..., :1])
  pixel_dirs = np.concatenate([pixel_coords + .5, homog], axis=-1)[..., None]
  cam_dirs = matmul(pix2cam, pixel_dirs)
  ray_dirs = matmul(cam2world[..., :3, :3], cam_dirs)[..., 0]
  ray_origins = np.broadcast_to(cam2world[..., :3, 3], ray_dirs.shape)
  return ray_origins, ray_dirs


def pix2cam_matrix(height, width, focal):
  return np.array([
      [1. / focal, 0, -.5 * width / focal],
      [0, -1. / focal, .5 * height / focal],
      [0, 0, -1.],
  ])


def camera_ray_batch(cam2world, hwf):
  height, width = int(hwf[0]), int(hwf[1])
  pix2cam = pix2cam_matrix(*hwf)
  pixel_coords = np.stack(
      np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
  return generate_rays(pixel_coords, pix2cam, cam2world)


def load_blender(data_dir, split):
  with open(
      os.path.join(data_dir, "transforms_{}.json".format(split)), "r") as fp:
    meta = json.load(fp)

  cams = []
  paths = []
  for i in range(len(meta["frames"])):
    frame = meta["frames"][i]
    cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
    fname = os.path.join(data_dir, frame["file_path"] + ".png")
    paths.append(fname)

  def image_read_fn(fname):
    with open(fname, "rb") as imgin:
      image = numpy.array(Image.open(imgin), dtype=numpy.float32) / 255.
    return image

  with ThreadPool() as pool:
    images = pool.map(image_read_fn, paths)
    pool.close()
    pool.join()

  images = numpy.stack(images, axis=0)
  images = images[..., :3]

  h, w = images.shape[1:3]
  camera_angle_x = float(meta["camera_angle_x"])
  focal = .5 * w / numpy.tan(.5 * camera_angle_x)

  hwf = numpy.array([h, w, focal], dtype=numpy.float32)
  poses = numpy.stack(cams, axis=0)
  poses = poses.astype(numpy.float32)
  poses[..., :3, 3] *= 0.033
  return {"images": images, "c2w": poses, "hwf": hwf}


def compute_ray_mesh_intersections(mesh, ray_origins, ray_directions):
  locations, index_ray, index_tri = mesh.ray.intersects_location(
      ray_origins, ray_directions, multiple_hits=False)
  return locations, index_ray, index_tri


def interpolate_uv(mesh, locations, triangle_indices):
  if mesh.visual.uv is None:
    raise ValueError("Mesh does not contain UV coordinates.")
  triangles = mesh.triangles[triangle_indices]
  tri_uv = mesh.visual.uv[mesh.faces[triangle_indices]]
  bary = trimesh.triangles.points_to_barycentric(triangles, locations)
  bary = numpy.clip(bary, 0.0, 1.0)
  bary = bary / numpy.maximum(
      numpy.sum(bary, axis=-1, keepdims=True), 1e-8)
  uv = (tri_uv * bary[..., None]).sum(axis=1)
  return uv


def preprocess():
  if scene_type != "synthetic":
    raise ValueError("preprocess_mesh_uv.py currently supports synthetic scenes.")

  data = load_blender(scene_dir, "train")
  images = data["images"]
  poses = data["c2w"]
  hwf = data["hwf"]

  mesh = trimesh.load(mesh_path, process=False)
  mesh = mesh.as_triangles()

  all_pixel_indices = []
  all_camera_indices = []
  all_uv = []
  all_viewdir = []
  all_color = []

  num_cams = poses.shape[0]
  h, w = images.shape[1:3]

  for cam_idx in range(num_cams):
    cam2world = poses[cam_idx, :3, :4]
    img = images[cam_idx]

    rays_o, rays_d = camera_ray_batch(cam2world, hwf)
    rays_o = numpy.array(rays_o).reshape(-1, 3)
    rays_d = numpy.array(rays_d).reshape(-1, 3)
    colors = img.reshape(-1, 3)

    locations, index_ray, index_tri = compute_ray_mesh_intersections(
        mesh, rays_o, rays_d)
    if locations.shape[0] == 0:
      continue

    uv = interpolate_uv(mesh, locations, index_tri)

    hit_dirs = rays_d[index_ray]
    hit_dirs = hit_dirs / numpy.linalg.norm(
        hit_dirs, axis=-1, keepdims=True).clip(1e-8, None)
    hit_colors = colors[index_ray]

    pixel_indices = index_ray.astype(numpy.int64)
    camera_indices = numpy.full_like(pixel_indices, cam_idx)

    all_pixel_indices.append(pixel_indices)
    all_camera_indices.append(camera_indices)
    all_uv.append(uv)
    all_viewdir.append(hit_dirs)
    all_color.append(hit_colors)

  if not all_uv:
    raise RuntimeError("No valid ray-mesh intersections were found.")

  pixel_indices = numpy.concatenate(all_pixel_indices, axis=0)
  camera_indices = numpy.concatenate(all_camera_indices, axis=0)
  uv = numpy.concatenate(all_uv, axis=0).astype(numpy.float32)
  viewdir = numpy.concatenate(all_viewdir, axis=0).astype(numpy.float32)
  color = numpy.concatenate(all_color, axis=0).astype(numpy.float32)

  numpy.savez_compressed(
      output_path,
      pixel_index=pixel_indices,
      camera_index=camera_indices,
      uv=uv,
      viewdir=viewdir,
      color=color,
      image_height=h,
      image_width=w)


if __name__ == "__main__":
  preprocess()

