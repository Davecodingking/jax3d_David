import argparse
import numpy as np
import trimesh
import xatlas


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) != 1:
            raise RuntimeError("Expected single mesh in OBJ")
        mesh = next(iter(mesh.geometry.values()))
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sponza_gt.obj")
    parser.add_argument("--output", default="sponza_gt_unwarpped.obj")
    args = parser.parse_args()

    mesh = load_mesh(args.input)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    print(f"Vertex count before: {len(vertices)}")

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    new_vertices = vertices[vmapping]
    new_faces = indices.reshape((-1, 3))

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        new_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)[vmapping]
    else:
        new_normals = None

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        vertex_normals=new_normals,
        process=False,
    )
    new_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)

    print(f"Vertex count after: {len(new_vertices)}")

    new_mesh.export(args.output, include_normals=True)


if __name__ == "__main__":
    main()
