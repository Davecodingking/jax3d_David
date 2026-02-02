import argparse
import numpy as np
import trimesh
try:
    import fast_simplification
except ImportError:
    fast_simplification = None
import xatlas
import time
import json
import os

def load_mesh(path):
    print(f"⏳ Loading mesh from {path}...")
    t0 = time.time()
    # force='mesh' prevents loading as Scene, ensures we get a Trimesh object
    mesh = trimesh.load(path, process=False, force='mesh')
    print(f"✅ Loaded in {time.time() - t0:.2f}s. Vertices: {len(mesh.vertices)}")
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sponza_gt.obj")
    parser.add_argument("--output", default="sponza_gt_unwarpped.obj")
    # Increase default target faces to 1M as requested (skip simplification for most meshes)
    parser.add_argument("--target_faces", type=int, default=1000000, help="Target face count for simplification")
    args = parser.parse_args()

    t_start = time.time()
    
    # 1. Load Mesh
    mesh = load_mesh(args.input)
    
    # 2. Optimize Geometry (Merge Vertices)
    print("🔧 Optimizing geometry (merge_vertices)...")
    t0 = time.time()
    # digits_vertex=4 (0.1mm precision) is faster and merges more aggressively than default
    # merge_norm=True keeps hard edges
    mesh.merge_vertices(merge_norm=True, merge_tex=False, digits_vertex=4)
    print(f"✅ Merged in {time.time() - t0:.2f}s. Vertices: {len(mesh.vertices)}")

    # 3. Decimation (Simplification)
    # Only simplify if absolutely necessary
    if len(mesh.faces) > args.target_faces:
        print(f"⚠️ High face count ({len(mesh.faces)}). Simplifying to {args.target_faces}...")
        t0 = time.time()
        
        # Try fast-simplification first (much faster)
        simplified = False
        if fast_simplification is not None:
            try:
                # Calculate reduction ratio (0.0 to 1.0)
                # target_reduction is the fraction of faces to REMOVE
                # e.g. if we want 500k faces from 2M, we want to remove 1.5M, so reduction is 0.75
                reduction_amount = 1.0 - (args.target_faces / len(mesh.faces))
                
                if 0.0 < reduction_amount < 1.0:
                    print(f"⚡ Using fast-simplification (removing {reduction_amount:.1%} of faces)...")
                    # fast_simplification.simplify returns (points, faces)
                    new_v, new_f = fast_simplification.simplify(
                        mesh.vertices, mesh.faces, target_reduction=reduction_amount
                    )
                    mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
                    simplified = True
            except Exception as e:
                print(f"⚠️ fast-simplification failed: {e}. Falling back to trimesh...")

        # Fallback to trimesh built-in
        if not simplified:
            try:
                # Quadratic decimation is high quality but can be slow. 
                # Since we raised the threshold, this only runs for massive meshes.
                try:
                    mesh = mesh.simplify_quadric_decimation(args.target_faces)
                except AttributeError:
                    # Fallback for older trimesh versions or if binding missing
                    print("⚠️ simplify_quadric_decimation not found, trying simplify_quadratic_decimation...")
                    mesh = mesh.simplify_quadratic_decimation(args.target_faces)
            except Exception as e:
                print(f"⚠️ Simplification failed: {e}. Skipping simplification.")
                
        print(f"✅ Simplified in {time.time() - t0:.2f}s. Faces: {len(mesh.faces)}")
    else:
        print("ℹ️ Skipping simplification (face count within limit).")

    # 4. Prepare for Xatlas
    print("🔄 Preparing data for xatlas...")
    # Optimization: Use float32 and uint32 to save memory and bandwidth
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    
    # 5. Run Xatlas
    print("🚀 Running xatlas parametrization...")
    t0 = time.time()
    
    # Check xatlas version if available
    if hasattr(xatlas, "__version__"):
        print(f"ℹ️ xatlas version: {xatlas.__version__}")

    # High Performance Settings
    # Note: Standard xatlas-python uses ChartOptions/PackOptions objects, not dictionaries.
    try:
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 1  # ⚡ Critical speedup (default 4)
        chart_options.straightness_weight = 6.0
        chart_options.normal_seam_weight = 4.0
        chart_options.normal_deviation_weight = 2.0
        chart_options.roundness_weight = 0.01

        pack_options = xatlas.PackOptions()
        # Some bindings use bruteForce (C++), others might use brute_force (Pythonic). 
        # We try to set what works or fallback.
        if hasattr(pack_options, "bruteForce"):
            pack_options.bruteForce = False
        elif hasattr(pack_options, "brute_force"):
            pack_options.brute_force = False
            
        pack_options.resolution = 2048
        pack_options.padding = 1
        # blockAlign is common in C++, check existence
        if hasattr(pack_options, "blockAlign"):
            pack_options.blockAlign = True
        elif hasattr(pack_options, "block_align"):
            pack_options.block_align = True

        print(f"⚙️ Configured xatlas: max_iter={chart_options.max_iterations}, resolution={pack_options.resolution}")
        
        # NOTE: xatlas.parametrize signature is:
        # parametrize(positions, indices, normals=None, uvs=None)
        # It does NOT accept chart_options/pack_options directly!
        # To use options, we must use the Atlas object API.
        
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        atlas.generate(chart_options, pack_options)
        vmapping, indices, uvs = atlas[0]
        
    except (AttributeError, TypeError) as e:
        # Fallback if xatlas version doesn't support options classes or specific attributes
        # Or if parametrize signature doesn't accept options (TypeError)
        print(f"⚠️ Warning: Could not configure advanced xatlas options ({e}). Using defaults.")
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    print(f"✅ Xatlas finished in {time.time() - t0:.2f}s")

    # 6. Reconstruct Mesh
    print("🔨 Reconstructing mesh...")
    new_vertices = vertices[vmapping]
    new_faces = indices.reshape((-1, 3))

    if mesh.vertex_normals is not None:
        # Map original normals to preserve hard edges
        # Note: We must cast to float32 consistent with vertices
        # However, indices from xatlas might refer to vertices that don't map 1:1 if original mesh was simplified.
        # Actually, 'vmapping' maps new vertices to ORIGINAL vertices (before xatlas, but AFTER simplification).
        # So we can just map the normals from the simplified mesh.
        try:
            new_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)[vmapping]
        except Exception as e:
            print(f"⚠️ Warning: Could not map normals ({e}). Recalculating...")
            new_normals = None
    else:
        new_normals = None

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        vertex_normals=new_normals,
        process=False,
    )
    
    # If normals are missing, trimesh might try to calculate them on export using scipy.
    # To avoid scipy dependency crash, we can try to compute them via trimesh internal methods
    # but ONLY if scipy is available, otherwise skip or accept basic normals.
    # Actually, trimesh.export(include_normals=True) will trigger calculation if vertex_normals is None.
    
    new_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)

    # 7. Export
    print(f"💾 Exporting to {args.output}...")
    new_mesh.export(args.output, include_normals=True)
    
    total_time = time.time() - t_start
    print(f"✨ All done in {total_time:.2f}s!")

if __name__ == "__main__":
    main()
