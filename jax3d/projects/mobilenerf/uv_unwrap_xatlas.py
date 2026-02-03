import argparse
import numpy as np
import trimesh
import tempfile

# Try to import pymeshlab for safe simplification
try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    pymeshlab = None
    HAS_PYMESHLAB = False

# Fallback to fast_simplification (less safe, but faster)
try:
    import fast_simplification
    HAS_FAST_SIMPLIFICATION = True
except ImportError:
    fast_simplification = None
    HAS_FAST_SIMPLIFICATION = False

import xatlas
import time
import os


def load_mesh(path):
    print(f"Loading mesh from {path}...")
    t0 = time.time()
    # force='mesh' prevents loading as Scene, ensures we get a Trimesh object
    mesh = trimesh.load(path, process=False, force='mesh')
    print(f"Loaded in {time.time() - t0:.2f}s. Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    return mesh


def validate_mesh(mesh, stage_name=""):
    """Basic validation to detect broken geometry."""
    issues = []

    # Check for degenerate faces (zero area)
    try:
        areas = mesh.area_faces
        degenerate_count = np.sum(areas < 1e-10)
        if degenerate_count > 0:
            issues.append(f"{degenerate_count} degenerate faces (zero area)")
    except Exception:
        pass

    # Check for invalid face indices
    max_vertex_idx = len(mesh.vertices) - 1
    if np.any(mesh.faces > max_vertex_idx) or np.any(mesh.faces < 0):
        issues.append("Invalid face indices detected")

    # Check for NaN/Inf in vertices
    if np.any(~np.isfinite(mesh.vertices)):
        issues.append("NaN or Inf values in vertices")

    if issues:
        print(f"[{stage_name}] Mesh validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"[{stage_name}] Mesh validation passed.")
        return True


def simplify_with_pymeshlab(mesh, target_faces, quality_threshold=0.3):
    """
    Safe simplification using pymeshlab (MeshLab algorithms).
    Preserves boundaries, normals, and topology.

    Args:
        mesh: trimesh.Trimesh object
        target_faces: target number of faces
        quality_threshold: 0-1, lower = stricter quality check (fewer collapses allowed)
    """
    print(f"Using pymeshlab for safe simplification (target: {target_faces} faces, quality_thr: {quality_threshold})...")

    # Save mesh to temp file (pymeshlab needs file input)
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
        tmp_input = tmp.name
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
        tmp_output = tmp.name

    try:
        # Export current mesh
        mesh.export(tmp_input)

        # Load into pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_input)

        print(f"  Before: {ms.current_mesh().face_number()} faces")

        # Safe decimation with all protections enabled
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,      # Keep mesh boundaries intact
            preservenormal=True,        # Prevent triangle flips
            preservetopology=True,      # Maintain mesh connectivity
            optimalplacement=True,      # Better vertex positioning
            planarquadric=True,         # Better handling of planar regions
            qualitythr=quality_threshold,  # Reject poor quality collapses (0-1, lower=stricter)
            autoclean=True,             # Remove unreferenced vertices
        )

        print(f"  After: {ms.current_mesh().face_number()} faces")

        # Save result
        ms.save_current_mesh(tmp_output)

        # Load back into trimesh
        simplified_mesh = trimesh.load(tmp_output, process=False, force='mesh')

        return simplified_mesh

    finally:
        # Cleanup temp files
        try:
            os.unlink(tmp_input)
            os.unlink(tmp_output)
        except Exception:
            pass


def simplify_with_fast_simplification(mesh, target_faces):
    """
    Fallback simplification using fast-simplification.
    Less safe but faster. Use only if pymeshlab unavailable.
    """
    print(f"Using fast-simplification (target: {target_faces} faces)...")
    print("  Warning: This method may cause geometry issues. Consider installing pymeshlab.")

    current_faces = len(mesh.faces)
    reduction_amount = 1.0 - (target_faces / current_faces)

    if not (0.0 < reduction_amount < 1.0):
        print("  Reduction amount out of range, skipping.")
        return mesh

    print(f"  Removing {reduction_amount:.1%} of faces...")
    new_v, new_f = fast_simplification.simplify(
        mesh.vertices.astype(np.float64),
        mesh.faces.astype(np.uint32),
        target_reduction=reduction_amount
    )

    return trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)


def main():
    parser = argparse.ArgumentParser(description="UV unwrap mesh using xatlas with optional simplification")
    parser.add_argument("--input", default="sponza_gt.obj", help="Input OBJ file")
    parser.add_argument("--output", default="sponza_gt_unwarpped.obj", help="Output OBJ file")
    parser.add_argument("--target_faces", type=int, default=1000000,
                        help="Target face count for simplification (default: 1000000)")
    parser.add_argument("--skip_simplification", action="store_true",
                        help="Skip simplification entirely, only do UV unwrapping")
    parser.add_argument("--force_fast_simplification", action="store_true",
                        help="Use fast-simplification instead of pymeshlab (not recommended)")
    parser.add_argument("--quality_threshold", type=float, default=0.3,
                        help="Quality threshold for decimation (0-1, lower=stricter, default: 0.3)")
    args = parser.parse_args()

    t_start = time.time()

    # Print available backends
    print("=" * 60)
    print("UV Unwrap Script (with safe simplification)")
    print("=" * 60)
    print(f"  pymeshlab available: {HAS_PYMESHLAB}")
    print(f"  fast-simplification available: {HAS_FAST_SIMPLIFICATION}")
    print("=" * 60)

    # 1. Load Mesh
    mesh = load_mesh(args.input)
    validate_mesh(mesh, "After load")

    # 2. Optimize Geometry (Merge Vertices)
    print("\nOptimizing geometry (merge_vertices)...")
    t0 = time.time()
    original_verts = len(mesh.vertices)
    # digits_vertex=5 (0.01mm precision) - less aggressive to preserve detail
    # merge_norm=True keeps hard edges
    mesh.merge_vertices(merge_norm=True, merge_tex=False, digits_vertex=5)
    print(f"Merged in {time.time() - t0:.2f}s. Vertices: {original_verts} -> {len(mesh.vertices)}")
    validate_mesh(mesh, "After merge")

    # 3. Decimation (Simplification)
    if args.skip_simplification:
        print("\nSimplification skipped (--skip_simplification flag set).")
    elif len(mesh.faces) <= args.target_faces:
        print(f"\nSkipping simplification (face count {len(mesh.faces)} <= target {args.target_faces}).")
    else:
        print(f"\nSimplifying mesh: {len(mesh.faces)} -> {args.target_faces} faces...")
        t0 = time.time()

        # Choose simplification backend
        if HAS_PYMESHLAB and not args.force_fast_simplification:
            try:
                mesh = simplify_with_pymeshlab(mesh, args.target_faces, args.quality_threshold)
            except Exception as e:
                print(f"pymeshlab failed: {e}")
                if HAS_FAST_SIMPLIFICATION:
                    print("Falling back to fast-simplification...")
                    mesh = simplify_with_fast_simplification(mesh, args.target_faces)
                else:
                    print("No fallback available. Skipping simplification.")
        elif HAS_FAST_SIMPLIFICATION:
            mesh = simplify_with_fast_simplification(mesh, args.target_faces)
        else:
            print("No simplification backend available. Skipping.")
            print("Install pymeshlab: pip install pymeshlab")

        print(f"Simplified in {time.time() - t0:.2f}s. Final faces: {len(mesh.faces)}")
        validate_mesh(mesh, "After simplification")

    # 4. Prepare for Xatlas
    print("\nPreparing data for xatlas...")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    # 5. Run Xatlas
    print("Running xatlas UV parametrization...")
    t0 = time.time()

    if hasattr(xatlas, "__version__"):
        print(f"  xatlas version: {xatlas.__version__}")

    # Configure xatlas options for speed while maintaining quality
    try:
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 1  # Speed optimization (default is 4)
        chart_options.straightness_weight = 6.0
        chart_options.normal_seam_weight = 4.0
        chart_options.normal_deviation_weight = 2.0
        chart_options.roundness_weight = 0.01

        pack_options = xatlas.PackOptions()
        # Handle different xatlas bindings (C++ vs Pythonic naming)
        if hasattr(pack_options, "bruteForce"):
            pack_options.bruteForce = False
        elif hasattr(pack_options, "brute_force"):
            pack_options.brute_force = False

        pack_options.resolution = 2048
        pack_options.padding = 1

        if hasattr(pack_options, "blockAlign"):
            pack_options.blockAlign = True
        elif hasattr(pack_options, "block_align"):
            pack_options.block_align = True

        print(f"  xatlas config: max_iterations={chart_options.max_iterations}, resolution={pack_options.resolution}")

        # Use Atlas API for full control
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        atlas.generate(chart_options, pack_options)
        vmapping, indices, uvs = atlas[0]

    except (AttributeError, TypeError) as e:
        print(f"  Warning: Advanced xatlas options unavailable ({e}). Using defaults.")
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    print(f"xatlas finished in {time.time() - t0:.2f}s")

    # 6. Reconstruct Mesh with UVs
    print("\nReconstructing mesh with UVs...")
    new_vertices = vertices[vmapping]
    new_faces = indices.reshape((-1, 3))

    # Preserve normals if available
    new_normals = None
    if mesh.vertex_normals is not None:
        try:
            new_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)[vmapping]
        except Exception as e:
            print(f"  Warning: Could not map normals ({e}). Will recalculate.")

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        vertex_normals=new_normals,
        process=False,
    )

    # Attach UV coordinates
    new_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)

    validate_mesh(new_mesh, "Final mesh")

    # 7. Export
    print(f"\nExporting to {args.output}...")
    new_mesh.export(args.output, include_normals=True)

    # Summary
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  Final vertices: {len(new_mesh.vertices)}")
    print(f"  Final faces: {len(new_mesh.faces)}")
    print(f"  Total time: {total_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
