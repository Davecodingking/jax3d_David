"""
PLY Mesh Loader with UV Coordinates for NeuralTextureWithLight.

This module provides functionality to load PLY meshes with embedded UV texture coordinates.
Supports both binary and ASCII PLY formats.

ADDED: New module for NeRF synthetic dataset support
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os


class PLYMesh:
    """
    A mesh loaded from a PLY file with vertices, faces, and UV coordinates.
    
    Attributes:
        vertices: Vertex positions as (N, 3) numpy array
        faces: Face indices as (M, 3) numpy array (triangles)
        uvs: UV coordinates as (N, 2) numpy array (per-vertex UVs)
        normals: Vertex normals as (N, 3) numpy array (optional)
        colors: Vertex colors as (N, 3) numpy array (optional)
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None
    ):
        """
        Initialize PLYMesh with geometry data.
        
        Args:
            vertices: Vertex positions (N, 3)
            faces: Triangle face indices (M, 3)
            uvs: UV texture coordinates (N, 2), optional
            normals: Vertex normals (N, 3), optional
            colors: Vertex colors (N, 3), optional
        """
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.normals = normals
        self.colors = colors
        
        # Validate shapes
        assert vertices.ndim == 2 and vertices.shape[1] == 3, \
            f"Vertices must be (N, 3), got {vertices.shape}"
        assert faces.ndim == 2 and faces.shape[1] == 3, \
            f"Faces must be (M, 3), got {faces.shape}"
        if uvs is not None:
            assert uvs.ndim == 2 and uvs.shape[1] == 2, \
                f"UVs must be (N, 2), got {uvs.shape}"
            assert uvs.shape[0] == vertices.shape[0], \
                f"UVs count ({uvs.shape[0]}) must match vertices count ({vertices.shape[0]})"
    
    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return self.vertices.shape[0]
    
    @property
    def num_faces(self) -> int:
        """Number of faces (triangles) in the mesh."""
        return self.faces.shape[0]
    
    @property
    def has_uvs(self) -> bool:
        """Whether the mesh has UV coordinates."""
        return self.uvs is not None
    
    @property
    def has_normals(self) -> bool:
        """Whether the mesh has vertex normals."""
        return self.normals is not None
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the mesh.
        
        Returns:
            Tuple of (min_corner, max_corner) as (3,) arrays
        """
        return self.vertices.min(axis=0), self.vertices.max(axis=0)
    
    def get_face_uvs(self) -> np.ndarray:
        """
        Get UV coordinates indexed by faces.
        
        Returns:
            UV coordinates for each face vertex as (M, 3, 2) array
        """
        if self.uvs is None:
            raise ValueError("Mesh does not have UV coordinates")
        return self.uvs[self.faces]
    
    def get_face_vertices(self) -> np.ndarray:
        """
        Get vertex positions indexed by faces.
        
        Returns:
            Vertex positions for each face as (M, 3, 3) array
        """
        return self.vertices[self.faces]
    
    def compute_face_normals(self) -> np.ndarray:
        """
        Compute face normals from vertex positions.
        
        Returns:
            Face normals as (M, 3) array
        """
        face_verts = self.get_face_vertices()  # (M, 3, 3)
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return normals / norms
    
    def __repr__(self) -> str:
        uv_str = f", uvs={self.uvs.shape}" if self.uvs is not None else ""
        return f"PLYMesh(vertices={self.vertices.shape}, faces={self.faces.shape}{uv_str})"


def _parse_ply_header(lines: List[str]) -> Dict[str, Any]:
    """
    Parse PLY file header to extract format and element information.
    
    Args:
        lines: List of header lines
    
    Returns:
        Dictionary with header information
    """
    header = {
        'format': 'ascii',
        'elements': [],
        'properties': {},
        'end_header_line': 0
    }
    
    current_element = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        keyword = parts[0]
        
        if keyword == 'ply':
            continue
        elif keyword == 'format':
            header['format'] = parts[1]  # 'ascii', 'binary_little_endian', etc.
        elif keyword == 'element':
            element_name = parts[1]
            element_count = int(parts[2])
            current_element = element_name
            header['elements'].append((element_name, element_count))
            header['properties'][element_name] = []
        elif keyword == 'property':
            if current_element is None:
                continue
            if parts[1] == 'list':
                # List property (e.g., vertex_indices for faces)
                prop_type = ('list', parts[2], parts[3], parts[4])
            else:
                prop_type = (parts[1], parts[2])
            header['properties'][current_element].append(prop_type)
        elif keyword == 'end_header':
            header['end_header_line'] = i
            break
    
    return header


def _get_numpy_dtype(ply_type: str) -> np.dtype:
    """Convert PLY type string to numpy dtype."""
    type_map = {
        'float': np.float32,
        'float32': np.float32,
        'double': np.float64,
        'float64': np.float64,
        'int': np.int32,
        'int32': np.int32,
        'uint': np.uint32,
        'uint32': np.uint32,
        'int8': np.int8,
        'uint8': np.uint8,
        'int16': np.int16,
        'uint16': np.uint16,
        'uchar': np.uint8,
        'char': np.int8,
        'short': np.int16,
        'ushort': np.uint16,
    }
    return type_map.get(ply_type, np.float32)


def load_ply(filepath: str) -> PLYMesh:
    """
    Load a PLY mesh file with UV coordinates.
    
    Supports ASCII and binary PLY formats. Looks for UV coordinates in
    properties named 's', 't' or 'texture_u', 'texture_v' or 'u', 'v'.
    
    Args:
        filepath: Path to the PLY file
    
    Returns:
        PLYMesh object with vertices, faces, and optionally UVs
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    
    Example:
        >>> mesh = load_ply("model.ply")
        >>> print(mesh.vertices.shape)  # (N, 3)
        >>> print(mesh.faces.shape)     # (M, 3)
        >>> if mesh.has_uvs:
        ...     print(mesh.uvs.shape)   # (N, 2)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PLY file not found: {filepath}")
    
    # Try using trimesh if available (more robust)
    try:
        import trimesh
        mesh = trimesh.load(filepath, process=False)
        
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        
        # Extract UV coordinates from visual
        uvs = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            uvs = np.array(mesh.visual.uv, dtype=np.float32)
        
        # Extract normals if available
        normals = None
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            normals = np.array(mesh.vertex_normals, dtype=np.float32)
        
        # Extract colors if available
        colors = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
        
        return PLYMesh(vertices, faces, uvs, normals, colors)
        
    except ImportError:
        pass  # Fall back to manual parsing
    
    # Manual PLY parsing (ASCII format)
    with open(filepath, 'rb') as f:
        # Read header (always ASCII)
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        header = _parse_ply_header(header_lines)
        
        # Find vertex and face elements
        vertex_count = 0
        face_count = 0
        for elem_name, elem_count in header['elements']:
            if elem_name == 'vertex':
                vertex_count = elem_count
            elif elem_name == 'face':
                face_count = elem_count
        
        vertex_props = header['properties'].get('vertex', [])
        
        # Find UV property indices
        uv_indices = {'u': -1, 'v': -1}
        uv_names = [('s', 't'), ('texture_u', 'texture_v'), ('u', 'v')]
        
        for i, prop in enumerate(vertex_props):
            if len(prop) >= 2:
                prop_name = prop[-1]
                for u_name, v_name in uv_names:
                    if prop_name == u_name:
                        uv_indices['u'] = i
                    elif prop_name == v_name:
                        uv_indices['v'] = i
        
        has_uvs = uv_indices['u'] >= 0 and uv_indices['v'] >= 0
        
        if header['format'] == 'ascii':
            # Read ASCII data
            vertices = []
            uvs = [] if has_uvs else None
            
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                values = [float(v) for v in line.split()]
                vertices.append(values[:3])
                if has_uvs:
                    uvs.append([values[uv_indices['u']], values[uv_indices['v']]])
            
            faces = []
            for _ in range(face_count):
                line = f.readline().decode('ascii').strip()
                values = [int(v) for v in line.split()]
                # First value is count, rest are indices
                faces.append(values[1:4])  # Assume triangles
            
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)
            uvs = np.array(uvs, dtype=np.float32) if uvs else None
            
        else:
            # Binary format
            raise NotImplementedError(
                f"Binary PLY format '{header['format']}' not implemented. "
                f"Please install trimesh for binary PLY support: pip install trimesh"
            )
    
    return PLYMesh(vertices, faces, uvs)


def load_ply_with_trimesh(filepath: str) -> PLYMesh:
    """
    Load PLY using trimesh library (more robust, handles more formats).
    
    Args:
        filepath: Path to the PLY file
    
    Returns:
        PLYMesh object
    
    Raises:
        ImportError: If trimesh is not installed
        FileNotFoundError: If file doesn't exist
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh library required for this function. "
            "Install with: pip install trimesh"
        )
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PLY file not found: {filepath}")
    
    mesh = trimesh.load(filepath, process=False)
    
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    # Extract UV coordinates
    uvs = None
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
        uvs = np.array(mesh.visual.uv, dtype=np.float32)
    
    # Extract normals
    normals = None
    if hasattr(mesh, 'vertex_normals'):
        normals = np.array(mesh.vertex_normals, dtype=np.float32)
    
    return PLYMesh(vertices, faces, uvs, normals)


def validate_mesh_for_rendering(mesh: PLYMesh) -> List[str]:
    """
    Validate that a mesh is suitable for UV rendering.
    
    Args:
        mesh: PLYMesh to validate
    
    Returns:
        List of warning/error messages (empty if valid)
    """
    issues = []
    
    if not mesh.has_uvs:
        issues.append("ERROR: Mesh does not have UV coordinates")
    else:
        # Check UV range
        uv_min = mesh.uvs.min()
        uv_max = mesh.uvs.max()
        if uv_min < 0.0 or uv_max > 1.0:
            issues.append(
                f"WARNING: UV coordinates outside [0, 1] range: [{uv_min:.3f}, {uv_max:.3f}]"
            )
    
    # Check for degenerate faces
    face_verts = mesh.get_face_vertices()
    areas = np.linalg.norm(
        np.cross(face_verts[:, 1] - face_verts[:, 0], 
                face_verts[:, 2] - face_verts[:, 0]), axis=1
    ) / 2
    degenerate_count = np.sum(areas < 1e-10)
    if degenerate_count > 0:
        issues.append(f"WARNING: {degenerate_count} degenerate (zero-area) faces detected")
    
    # Check face indices
    max_idx = mesh.faces.max()
    if max_idx >= mesh.num_vertices:
        issues.append(
            f"ERROR: Face indices ({max_idx}) exceed vertex count ({mesh.num_vertices})"
        )
    
    return issues
