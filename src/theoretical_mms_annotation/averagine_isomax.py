"""
Table-based isomax(FragmentMass) lookup (Supplemental Note 10).

Uses a table of representative Averagine isotopic distributions (~one entry
per 10 Da, from Basharat et al., 2023) to estimate, for a given fragment
mass, how many isotopic variants (+1, +2, ...) are needed before the
relative intensity (normalized so the most abundant peak = 100%) drops
below a cutoff t (default 5%, matching james_code/IsoMax_example.f90).

For a query FragmentMass we find the table entry whose monoisotopic mass is
closest, then take isomax to be the largest isotopic variant +i whose
relative intensity is >= t (variant +(isomax+1) is < t). Table entries are
unimodal in intensity, so this is simply the largest index anywhere in the
entry's envelope with intensity >= t.
"""

import bisect
from pathlib import Path

DEFAULT_TABLE_PATH = Path(__file__).parent / "james_code" / "Averagine_Envelopes"


def _load_table(path):
    """Parse the Averagine_Envelopes file into parallel lists.

    Returns (masses, intensities) where masses[i] is the monoisotopic mass
    of table entry i and intensities[i] is a {iso: relative_intensity} dict
    for that entry, already normalized so the most abundant peak is 100.

    A handful of entries in the table omit the iso=0 line (its intensity
    underflowed the file's print precision for very heavy masses); we key
    by the explicit iso label from each line rather than assuming a
    contiguous 0..num_iso-1 range, so gaps like this are handled correctly.
    """
    masses = []
    intensities = []
    with open(path) as f:
        blocks = f.read().strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        mass, num_iso = lines[0].split()
        num_iso = int(num_iso)
        intens = {}
        for line in lines[1:1 + num_iso]:
            iso, _iso_mass, inten = line.split()
            intens[int(iso)] = float(inten)
        masses.append(float(mass))
        intensities.append(intens)

    return masses, intensities


class AveragineIsoMaxTable:
    """Table-based isomax(FragmentMass) estimator."""

    def __init__(self, cutoff_pct=5.0, table_path=DEFAULT_TABLE_PATH):
        self.cutoff_pct = cutoff_pct
        self._masses, self._intensities = _load_table(table_path)
        self._isomax = [self._isomax_for_entry(i) for i in range(len(self._masses))]

    def _isomax_for_entry(self, entry_idx):
        intens = self._intensities[entry_idx]
        isomax = 0
        for iso, val in intens.items():
            if val >= self.cutoff_pct and iso > isomax:
                isomax = iso
        return isomax

    def _nearest_entry(self, fragment_mass):
        """Index of the table entry with monoisotopic mass closest to fragment_mass."""
        idx = bisect.bisect_left(self._masses, fragment_mass)
        if idx == 0:
            return 0
        if idx == len(self._masses):
            return len(self._masses) - 1
        before, after = self._masses[idx - 1], self._masses[idx]
        if abs(fragment_mass - before) <= abs(fragment_mass - after):
            return idx - 1
        return idx

    def get_isomax(self, fragment_mass):
        """Return isomax(fragment_mass): the highest isotopic variant to keep."""
        return self._isomax[self._nearest_entry(fragment_mass)]


if __name__ == "__main__":
    table = AveragineIsoMaxTable(cutoff_pct=5.0)
    for m in [500.0, 1500.0, 5000.0, 20000.0, 110000.0]:
        print(f"FragmentMass={m:>10.1f}  isomax={table.get_isomax(m)}")
