"""Module for checking consistency between mapped and unmapped data."""


def check_mapped_vs_unmapped(H, H_unmapped, print_rows=True):
    """Check that mapped and unmapped data are consistent."""
    # Compare mapped vs unmapped for first few rows
    if print_rows:
        print("\nMapped vs Unmapped comparison:")
        for i in range(min(3, len(H))):
            print(f"\nRow {i}:")
            print(
                f"  Mapped   - sim_id: {H['sim_id'][i]}, x: {H['x'][i]}, x_on_cube: {H['x_on_cube'][i]}"
            )
            print(
                f"  Unmapped - sim_id: {H_unmapped['sim_id'][i]}, "
                + f"beam_i_r2: {H_unmapped['beam_i_r2'][i]}, beam_z_i_2: {H_unmapped['beam_z_i_2'][i]}, "
                + f"beam_length: {H_unmapped['beam_length'][i]}, beam_i_r2_on_cube: {H_unmapped['beam_i_r2_on_cube'][i]}, "
                + f"beam_z_i_2_on_cube: {H_unmapped['beam_z_i_2_on_cube'][i]}, beam_length_on_cube: {H_unmapped['beam_length_on_cube'][i]}"
            )

    for i in range(len(H)):
        # Check sim_id matches
        assert (
            H["sim_id"][i] == H_unmapped["sim_id"][i]
        ), f"sim_id mismatch at row {i}"

        # Check x array matches individual variables
        assert (
            H["x"][i][0] == H_unmapped["beam_i_r2"][i]
        ), f"x[0] != beam_i_r2 at row {i}"
        assert (
            H["x"][i][1] == H_unmapped["beam_z_i_2"][i]
        ), f"x[1] != beam_z_i_2 at row {i}"
        assert (
            H["x"][i][2] == H_unmapped["beam_length"][i]
        ), f"x[2] != beam_length at row {i}"

        # Check x_on_cube array matches individual variables
        assert (
            H["x_on_cube"][i][0] == H_unmapped["beam_i_r2_on_cube"][i]
        ), f"x_on_cube[0] != beam_i_r2_on_cube at row {i}"
        assert (
            H["x_on_cube"][i][1] == H_unmapped["beam_z_i_2_on_cube"][i]
        ), f"x_on_cube[1] != beam_z_i_2_on_cube at row {i}"
        assert (
            H["x_on_cube"][i][2] == H_unmapped["beam_length_on_cube"][i]
        ), f"x_on_cube[2] != beam_length_on_cube at row {i}"

    print("\nMapped and unmapped data are consistent!")
