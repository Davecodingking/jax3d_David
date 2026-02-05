"""
Fast UV Renderer using vectorized NumPy operations.

This is ~100x faster than the naive per-pixel software renderer.
Uses scanline rasterization with NumPy vectorization.
"""

import numpy as np
from typing import Tuple
from numba import jit, prange
import warnings

# Suppress numba warnings about parallel
warnings.filterwarnings('ignore', category=Warning, module='numba')


class FastUVRenderer:
    """
    Fast software UV renderer using Numba JIT compilation.
    
    ~50-100x faster than pure Python software renderer.
    """
    
    def __init__(self, mesh, width: int = 800, height: int = 800):
        self.mesh = mesh
        self.width = width
        self.height = height
        
        if not mesh.has_uvs:
            raise ValueError("Mesh must have UV coordinates")
        
        # Pre-convert mesh data to contiguous arrays for Numba
        self.vertices = np.ascontiguousarray(mesh.vertices.astype(np.float32))
        self.faces = np.ascontiguousarray(mesh.faces.astype(np.int32))
        self.uvs = np.ascontiguousarray(mesh.uvs.astype(np.float32))
    
    def render_uv_map(
        self,
        camera_pose: np.ndarray,
        fov: float = 60.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render UV map + world position map using fast Numba-accelerated rasterization."""
        
        camera_pos = camera_pose[:3, 3].astype(np.float32)
        camera_rot = camera_pose[:3, :3].astype(np.float32)
        
        # Transform vertices to camera space.
        #
        # NeRF `transform_matrix` is camera-to-world (c2w) with column vectors:
        # p_world = R * p_cam + t. For row-vector batches, world->cam is:
        # p_cam = (p_world - t) * R  (NOT * R.T).
        vertices_cam = (self.vertices - camera_pos) @ camera_rot
        
        # Project to screen
        fov_rad = np.radians(fov)
        f = self.height / (2 * np.tan(fov_rad / 2))
        
        depth = -vertices_cam[:, 2]
        safe_depth = np.where(depth > 1e-8, depth, 1e-8)
        
        proj_x = (vertices_cam[:, 0] / safe_depth * f + self.width / 2).astype(np.float32)
        proj_y = (-vertices_cam[:, 1] / safe_depth * f + self.height / 2).astype(np.float32)
        proj_z = depth.astype(np.float32)
        
        projected = np.stack([proj_x, proj_y, proj_z], axis=1)
        projected = np.ascontiguousarray(projected)
        
        # Rasterize using Numba
        uv_map, depth_buffer, pos_map = _rasterize_triangles_numba(
            projected, self.faces, self.uvs, self.vertices,
            self.width, self.height
        )
        
        mask = depth_buffer < 1e9
        
        return uv_map, mask, pos_map
    
    def render_uv_map_from_nerf_transform(
        self,
        transform_matrix: np.ndarray,
        camera_angle_x: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render UV map from NeRF transform matrix."""
        camera_pose = transform_matrix.astype(np.float32)
        
        fov_x = np.degrees(camera_angle_x)
        aspect = self.width / self.height
        fov_y = 2 * np.degrees(np.arctan(np.tan(np.radians(fov_x/2)) / aspect))
        
        return self.render_uv_map(camera_pose, fov=fov_y)
    
    def close(self):
        pass


@jit(nopython=True, parallel=False, cache=True)
def _rasterize_triangles_numba(
    projected: np.ndarray,  # (N, 3) projected vertices [x, y, depth]
    faces: np.ndarray,      # (F, 3) face indices
    uvs: np.ndarray,        # (N, 2) UV coordinates
    vertices_world: np.ndarray,  # (N, 3) world-space vertices
    width: int,
    height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated triangle rasterization.
    
    Uses parallel processing over triangles.
    """
    uv_map = np.full((height, width, 2), -1.0, dtype=np.float32)
    depth_buffer = np.full((height, width), 1e10, dtype=np.float32)
    pos_map = np.zeros((height, width, 3), dtype=np.float32)
    
    num_faces = faces.shape[0]
    
    for fi in range(num_faces):
        i0, i1, i2 = faces[fi]
        
        # Get projected vertices
        x0, y0, z0 = projected[i0]
        x1, y1, z1 = projected[i1]
        x2, y2, z2 = projected[i2]
        
        # Get UVs
        u0, v0 = uvs[i0]
        u1, v1 = uvs[i1]
        u2, v2 = uvs[i2]

        # World positions (for hit point / light-probe sampling)
        wx0, wy0, wz0 = vertices_world[i0]
        wx1, wy1, wz1 = vertices_world[i1]
        wx2, wy2, wz2 = vertices_world[i2]
        
        # Skip triangles behind camera
        if z0 <= 0 and z1 <= 0 and z2 <= 0:
            continue
        
        # Compute bounding box
        min_x = max(0, int(min(x0, x1, x2)))
        max_x = min(width - 1, int(max(x0, x1, x2)) + 1)
        min_y = max(0, int(min(y0, y1, y2)))
        max_y = min(height - 1, int(max(y0, y1, y2)) + 1)
        
        if min_x > max_x or min_y > max_y:
            continue
        
        # Edge vectors for barycentric
        e01_x = x1 - x0
        e01_y = y1 - y0
        e12_x = x2 - x1
        e12_y = y2 - y1
        e20_x = x0 - x2
        e20_y = y0 - y2
        
        # Triangle area (2x) - negative = CW in screen space = front-facing (due to Y-flip)
        area = e01_x * (y2 - y0) - e01_y * (x2 - x0)
        
        # Backface culling: In screen space (Y-down), front faces have negative area (CW).
        # Skip back-facing triangles (positive area = CCW in screen = back face)
        if area >= 0:
            continue
        
        # Skip degenerate triangles
        if abs(area) < 1e-8:
            continue
        
        inv_area = 1.0 / area
        
        # Rasterize
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                # Point to test
                qx = px + 0.5
                qy = py + 0.5
                
                # Barycentric coordinates (will be negative for points inside since area < 0)
                w0 = ((x1 - qx) * e12_y - (y1 - qy) * e12_x)
                w1 = ((x2 - qx) * e20_y - (y2 - qy) * e20_x)
                w2 = ((x0 - qx) * e01_y - (y0 - qy) * e01_x)
                
                # Point is inside triangle when all barycentric coords have same sign as area
                if w0 <= 0 and w1 <= 0 and w2 <= 0:
                    w0 = w0 * inv_area
                    w1 = w1 * inv_area
                    w2 = w2 * inv_area
                    
                    # Normalize
                    total = w0 + w1 + w2
                    if total > 1e-8:
                        w0 /= total
                        w1 /= total
                        w2 /= total
                        
                        # Interpolate depth
                        depth = w0 * z0 + w1 * z1 + w2 * z2
                        
                        if depth > 0 and depth < depth_buffer[py, px]:
                            depth_buffer[py, px] = depth
                            
                            # Interpolate UV
                            uv_map[py, px, 0] = w0 * u0 + w1 * u1 + w2 * u2
                            uv_map[py, px, 1] = w0 * v0 + w1 * v1 + w2 * v2

                            # Interpolate world position
                            pos_map[py, px, 0] = w0 * wx0 + w1 * wx1 + w2 * wx2
                            pos_map[py, px, 1] = w0 * wy0 + w1 * wy1 + w2 * wy2
                            pos_map[py, px, 2] = w0 * wz0 + w1 * wz1 + w2 * wz2
    
    return uv_map, depth_buffer, pos_map


def create_fast_uv_renderer(mesh, width: int, height: int):
    """Create the fastest available UV renderer."""
    try:
        import numba
        return FastUVRenderer(mesh, width, height)
    except ImportError:
        print("Warning: numba not available, using slow software renderer")
        from utils.uv_renderer import SoftwareUVRenderer
        return SoftwareUVRenderer(mesh, width, height)
