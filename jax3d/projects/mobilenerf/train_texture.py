import os
import json
import numpy
import cv2
import jax
import jax.numpy as np
from jax import random
import flax
import flax.linen as nn
from typing import Sequence


scene_type = "synthetic"
object_name = "MyNeRFData"
scene_dir = "data/custom/" + object_name
uv_samples_path = os.path.join(scene_dir, "mesh_uv_samples.npz")
weights_dir = "weights_texture"
textures_dir = "textures"


num_bottleneck_features = 8
features_per_texture = num_bottleneck_features // 2
texture_size = 2048
batch_size = 4096
training_iters = 200000
learning_rate = 1e-3


if not os.path.exists(weights_dir):
  os.makedirs(weights_dir)
if not os.path.exists(textures_dir):
  os.makedirs(textures_dir)


class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


def bilinear_sample(texture, uv):
  h, w, c = texture.shape
  u = uv[..., 0] * (w - 1)
  v = (1.0 - uv[..., 1]) * (h - 1)

  x0 = np.floor(u).astype(np.int32)
  x1 = np.clip(x0 + 1, 0, w - 1)
  y0 = np.floor(v).astype(np.int32)
  y1 = np.clip(y0 + 1, 0, h - 1)

  x0 = np.clip(x0, 0, w - 1)
  y0 = np.clip(y0, 0, h - 1)

  wa = (x1 - u) * (y1 - v)
  wb = (u - x0) * (y1 - v)
  wc = (x1 - u) * (v - y0)
  wd = (u - x0) * (v - y0)

  ia = texture[y0, x0]
  ib = texture[y0, x1]
  ic = texture[y1, x0]
  id = texture[y1, x1]

  return (wa[..., None] * ia + wb[..., None] * ib +
          wc[..., None] * ic + wd[..., None] * id)


def render_batch(params, uv, viewdir, color_model):
  tex1, tex2, color_vars = params
  f1 = bilinear_sample(tex1, uv)
  f2 = bilinear_sample(tex2, uv)
  features = np.concatenate([f1, f2], axis=-1)
  inputs = np.concatenate([features, viewdir], axis=-1)
  rgb = jax.nn.sigmoid(color_model.apply(color_vars, inputs))
  return rgb


def make_dataset():
  data = numpy.load(uv_samples_path)
  uv = np.array(data["uv"])
  viewdir = np.array(data["viewdir"])
  color = np.array(data["color"])
  pixel_index = numpy.array(data["pixel_index"])
  camera_index = numpy.array(data["camera_index"])
  h = int(data["image_height"])
  w = int(data["image_width"])
  return uv, viewdir, color, pixel_index, camera_index, h, w


def loss_fn(params, batch, color_model):
  uv, viewdir, gt_color = batch
  pred = render_batch(params, uv, viewdir, color_model)
  loss = np.mean((pred - gt_color) ** 2)
  return loss


def compute_texel_mask(uv):
  h = texture_size
  w = texture_size
  u = uv[:, 0] * (w - 1)
  v = (1.0 - uv[:, 1]) * (h - 1)
  x = numpy.rint(u).astype(numpy.int32)
  y = numpy.rint(v).astype(numpy.int32)
  x = numpy.clip(x, 0, w - 1)
  y = numpy.clip(y, 0, h - 1)
  mask = numpy.zeros((h, w), dtype=numpy.uint8)
  mask[y, x] = 1
  return mask


def dilate_texture(texture, mask, iterations=10):
  tex = texture.copy()
  m = mask.copy()
  h, w, c = tex.shape
  for _ in range(iterations):
    new_tex = numpy.zeros_like(tex)
    new_mask = numpy.zeros_like(m)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      shifted_tex = numpy.roll(tex, shift=(dy, dx), axis=(0, 1))
      shifted_mask = numpy.roll(m, shift=(dy, dx), axis=(0, 1))
      update = (m == 0) & (shifted_mask > 0)
      update_idx = numpy.where(update)
      new_tex[update_idx] += shifted_tex[update_idx]
      new_mask[update_idx] += 1
    update_pixels = new_mask > 0
    if not numpy.any(update_pixels):
      break
    tex[update_pixels] = (
        new_tex[update_pixels] /
        new_mask[update_pixels][..., None].astype(numpy.float32))
    m[update_pixels] = 1
  return tex


def save_texture(path, tex):
  img = numpy.clip(numpy.array(tex), 0.0, 1.0).astype(numpy.float32)
  cv2.imwrite(path, img[..., ::-1])


def save_png(path, img):
  arr = numpy.clip(numpy.array(img) * 255.0, 0.0, 255.0).astype(
      numpy.uint8)
  cv2.imwrite(path, arr[..., ::-1])


def tree_to_json_dict(tree):
  if isinstance(tree, dict):
    return {k: tree_to_json_dict(v) for k, v in tree.items()}
  arr = numpy.array(tree)
  return {
      "shape": list(arr.shape),
      "values": arr.reshape(-1).tolist(),
  }


def train():
  if scene_type != "synthetic":
    raise ValueError("train_texture.py currently supports synthetic scenes.")

  uv_all_np, viewdir_all_np, color_all_np, pixel_index_all, camera_index_all, img_h, img_w = make_dataset()
  num_samples = uv_all_np.shape[0]

  color_model = MLP([16, 16, 3])
  dummy_in = np.zeros((1, num_bottleneck_features + 3), np.float32)
  color_vars = color_model.init(random.PRNGKey(0), dummy_in)

  key = random.PRNGKey(0)
  key, k1, k2 = random.split(key, 3)
  tex1 = random.uniform(
      k1,
      (texture_size, texture_size, features_per_texture),
      minval=0.0,
      maxval=0.1)
  tex2 = random.uniform(
      k2,
      (texture_size, texture_size, features_per_texture),
      minval=0.0,
      maxval=0.1)

  params = [tex1, tex2, color_vars]
  optimizer = flax.optim.Adam(learning_rate).create(params)

  uv_all = jax.device_put(uv_all_np)
  viewdir_all = jax.device_put(viewdir_all_np)
  color_all = jax.device_put(color_all_np)

  @jax.jit
  def train_step(opt, rng):
    rng, rng_idx = random.split(rng)
    idx = random.randint(rng_idx, (batch_size,), 0, num_samples)
    batch_uv = uv_all[idx]
    batch_viewdir = viewdir_all[idx]
    batch_color = color_all[idx]

    def _loss_fn(p):
      return loss_fn(p, (batch_uv, batch_viewdir, batch_color), color_model)

    loss, grads = jax.value_and_grad(_loss_fn)(opt.target)
    opt = opt.apply_gradient(grads)
    return opt, loss, rng

  for i in range(training_iters):
    optimizer, loss_value, key = train_step(optimizer, key)
    if (i + 1) % 1000 == 0:
      print("iter", i + 1, "loss", float(loss_value))

  trained_params = optimizer.target
  cam_id = int(camera_index_all[0])
  cam_mask = camera_index_all == cam_id
  uv_cam = np.array(uv_all_np[cam_mask])
  viewdir_cam = np.array(viewdir_all_np[cam_mask])
  pix_idx_cam = pixel_index_all[cam_mask]
  gt_cam = color_all_np[cam_mask]
  preds_cam = render_batch(trained_params, uv_cam, viewdir_cam, color_model)
  preds_cam_np = numpy.array(preds_cam)
  img_pred = numpy.zeros((img_h, img_w, 3), numpy.float32)
  img_gt = numpy.zeros((img_h, img_w, 3), numpy.float32)
  y = pix_idx_cam // img_w
  x = pix_idx_cam % img_w
  img_pred[y, x] = preds_cam_np
  img_gt[y, x] = gt_cam
  tex1_trained = numpy.array(trained_params[0])
  tex2_trained = numpy.array(trained_params[1])
  color_vars_trained = trained_params[2]

  mask = compute_texel_mask(uv_all_np)
  tex1_dilated = dilate_texture(tex1_trained, mask)
  tex2_dilated = dilate_texture(tex2_trained, mask)

  save_texture(os.path.join(textures_dir, "texture1.exr"), tex1_dilated)
  save_texture(os.path.join(textures_dir, "texture2.exr"), tex2_dilated)

  save_png(os.path.join(textures_dir, "preview_cam0_pred.png"), img_pred)
  save_png(os.path.join(textures_dir, "preview_cam0_gt.png"), img_gt)

  weights_json = tree_to_json_dict(color_vars_trained)
  with open(os.path.join(weights_dir, "color_mlp.json"), "w") as f:
    json.dump(weights_json, f)


if __name__ == "__main__":
  train()
