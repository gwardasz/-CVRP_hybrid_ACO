import math
import numpy as np

class CVRPInstance:
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = ""
        self.n_locations = 0
        self.capacity = 0
        self.coords = []   # List of (x, y) tuples
        self.demands = []  # List of integers
        self.dist_matrix = None
        self.bks = None

        # Automatically load and process data upon initialization
        self._read_file()
        self._compute_distance_matrix()

    def _read_file(self):
        """
        Parses a standard VRP text file.
        Assumes format often found in Christofides (CMT) sets:
        - Metadata headers
        - SECTION 'NODE_COORD_SECTION' (id, x, y)
        - SECTION 'DEMAND_SECTION' (id, demand)
        - SECTION 'DEPOT_SECTION' (id)
        """
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        section = None
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "NAME":
                self.name = parts[-1]
            # --- NEW: Parse COMMENT for BKS ---
            elif parts[0] == "COMMENT":
                # CMT files format: "COMMENT : 524.61" -> parts[-1] is the value
                try:
                    # Remove any non-numeric characters just in case, or take last element
                    val_str = parts[-1] 
                    self.bks = float(val_str)
                except ValueError:
                    self.bks = None # Could not parse BKS
            # ----------------------------------
            elif parts[0] == "DIMENSION":
                self.n_locations = int(parts[-1])
            elif parts[0] == "CAPACITY":
                self.capacity = int(parts[-1])
            elif parts[0] == "NODE_COORD_SECTION":
                section = "COORD"
                continue
            elif parts[0] == "DEMAND_SECTION":
                section = "DEMAND"
                continue
            elif parts[0] == "DEPOT_SECTION":
                section = "DEPOT"
                continue
            elif parts[0] == "EOF":
                break

            if section == "COORD":
                self.coords.append((float(parts[1]), float(parts[2])))
            elif section == "DEMAND":
                self.demands.append(int(parts[1]))

        self.demands = np.array(self.demands)

    def _compute_distance_matrix(self):
        """
        Calculates the Euclidean distance matrix for all node pairs.
        """
        n = self.n_locations
        self.dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.coords[i]
                    x2, y2 = self.coords[j]
                    # Euclidean Distance Formula
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    self.dist_matrix[i][j] = dist

    def get_data(self):
        """Returns the essential data structures for the Solver."""
        return self.dist_matrix, self.demands, self.capacity