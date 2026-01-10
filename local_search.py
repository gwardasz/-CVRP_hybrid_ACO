import numpy as np

class LocalSearch:
    def __init__(self, dist_matrix, demands, capacity):
        """
        Initialized with problem constraints to ensure valid moves.
        """
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity

    def optimize_solution(self, full_tour):
        """
        VND Main Loop:
        1. Relocate -> 2. Swap -> 3. 2-Opt (Intra) -> 4. 2-Opt* (Inter)
        Repeats until no operator can find an improvement.
        """
        # 1. Parse flat tour into separate routes: [[1,2], [3,4], ...]
        routes = self._extract_routes(full_tour)
        
        improvement = True
        while improvement:
            improvement = False
            
            # --- Operator 1: Relocate (Move customer to different route) ---
            # [cite: 1390]
            if self._relocate_move(routes):
                improvement = True
                continue # Restart VND from top
                
            # --- Operator 2: Swap (Exchange customers between routes) ---
            # [cite: 1392]
            if self._swap_move(routes):
                improvement = True
                continue # Restart VND from top
                
            # --- Operator 3: 2-Opt (Intra-route uncrossing) ---
            # [cite: 1399]
            if self._two_opt_intra(routes):
                improvement = True
                continue
                
            # --- Operator 4: 2-Opt* (Inter-route tail swap) ---
            # 
            if self._two_opt_star(routes):
                improvement = True
                continue

        # 2. Reconstruct flat tour
        final_tour = [0]
        total_cost = 0.0
        
        for r in routes:
            if not r: continue # Skip empty routes
            final_tour.extend(r)
            final_tour.append(0)
            total_cost += self._calculate_route_cost(r)
            
        return final_tour, total_cost

    # =========================================================================
    #  OPERATOR 1: RELOCATE (Move 1 node)
    # =========================================================================
    def _relocate_move(self, routes):
        best_gain = 0
        move = None
        
        # Iterate all routes and nodes
        for r_idx_src, src_route in enumerate(routes):
            for i in range(len(src_route)):
                node = src_route[i]
                load = self.demands[node]
                
                # Try inserting into every other route
                for r_idx_dst, dst_route in enumerate(routes):
                    if r_idx_src == r_idx_dst: continue # Simple inter-route only
                    
                    # Capacity Check
                    if self._get_route_load(dst_route) + load > self.capacity:
                        continue
                        
                    # Try all positions in dest route
                    # 0 to len because we can insert at end
                    for j in range(len(dst_route) + 1):
                        gain = self._calc_relocate_gain(src_route, dst_route, i, j)
                        if gain > best_gain:
                            best_gain = gain
                            move = (r_idx_src, r_idx_dst, i, j)
        
        if move:
            r_src, r_dst, i, j = move
            node = routes[r_src].pop(i)
            routes[r_dst].insert(j, node)
            return True
        return False

    def _calc_relocate_gain(self, r_src, r_dst, i, j):
        # Cost removed from Src
        A = r_src[i-1] if i > 0 else 0
        B = r_src[i]
        C = r_src[i+1] if i < len(r_src)-1 else 0
        removed = self.dist_matrix[A][B] + self.dist_matrix[B][C] - self.dist_matrix[A][C]
        
        # Cost added to Dst
        F = r_dst[j-1] if j > 0 else 0
        G = r_dst[j] if j < len(r_dst) else 0 # 0 if inserting at end
        added = self.dist_matrix[F][B] + self.dist_matrix[B][G] - self.dist_matrix[F][G]
        
        return removed - added

    # =========================================================================
    #  OPERATOR 2: SWAP (Exchange 1-1)
    # =========================================================================
    def _swap_move(self, routes):
        best_gain = 0
        move = None
        
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)): # Avoid duplicates
                r1 = routes[r1_idx]
                r2 = routes[r2_idx]
                
                load1 = self._get_route_load(r1)
                load2 = self._get_route_load(r2)
                
                for i in range(len(r1)):
                    for j in range(len(r2)):
                        node1 = r1[i]
                        node2 = r2[j]
                        
                        # Capacity Check: New loads must be valid
                        new_load1 = load1 - self.demands[node1] + self.demands[node2]
                        new_load2 = load2 - self.demands[node2] + self.demands[node1]
                        
                        if new_load1 > self.capacity or new_load2 > self.capacity:
                            continue
                            
                        gain = self._calc_swap_gain(r1, r2, i, j)
                        if gain > best_gain:
                            best_gain = gain
                            move = (r1_idx, r2_idx, i, j)
                            
        if move:
            r1, r2, i, j = move
            # Swap
            routes[r1][i], routes[r2][j] = routes[r2][j], routes[r1][i]
            return True
        return False

    def _calc_swap_gain(self, r1, r2, i, j):
        # Neighbors of Node 1
        A = r1[i-1] if i > 0 else 0
        B = r1[i]
        C = r1[i+1] if i < len(r1)-1 else 0
        
        # Neighbors of Node 2
        U = r2[j-1] if j > 0 else 0
        V = r2[j]
        W = r2[j+1] if j < len(r2)-1 else 0
        
        # Old Edges Cost
        old_cost = (self.dist_matrix[A][B] + self.dist_matrix[B][C] +
                    self.dist_matrix[U][V] + self.dist_matrix[V][W])
        
        # New Edges Cost (B is now at V's spot, V is at B's spot)
        new_cost = (self.dist_matrix[A][V] + self.dist_matrix[V][C] +
                    self.dist_matrix[U][B] + self.dist_matrix[B][W])
                    
        return old_cost - new_cost

    # =========================================================================
    #  OPERATOR 3: 2-OPT (Intra-Route)
    # =========================================================================
    def _two_opt_intra(self, routes):
        # This is your existing logic, applied to each route
        improvement_found = False
        for route in routes:
            if len(route) < 2: continue
            
            # Simple First-Improvement for speed
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        if j - i == 1: continue 
                        
                        # Check gain
                        A = route[i-1] if i > 0 else 0
                        B = route[i]
                        C = route[j]
                        D = route[j+1] if j < len(route)-1 else 0
                        
                        curr_dist = self.dist_matrix[A][B] + self.dist_matrix[C][D]
                        new_dist = self.dist_matrix[A][C] + self.dist_matrix[B][D]
                        
                        if new_dist < curr_dist:
                            route[i:j+1] = route[i:j+1][::-1]
                            improved = True
                            improvement_found = True
        return improvement_found

    # =========================================================================
    #  OPERATOR 4: 2-OPT* (Inter-Route Tail Swap)
    # =========================================================================
    def _two_opt_star(self, routes):
        """
        Cuts two routes and swaps their tails.
        Route 1: Start -> i -> (tail1)
        Route 2: Start -> j -> (tail2)
        Result 1: Start -> i -> (tail2)
        Result 2: Start -> j -> (tail1)
        """
        best_gain = 0
        move = None
        
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)):
                r1 = routes[r1_idx]
                r2 = routes[r2_idx]
                
                # Try all break points
                # i is last node of first part of r1
                # j is last node of first part of r2
                for i in range(len(r1)):
                    for j in range(len(r2)):
                        # Load of HEAD parts
                        load1_head = sum(self.demands[n] for n in r1[:i+1])
                        load2_head = sum(self.demands[n] for n in r2[:j+1])
                        
                        # Load of TAIL parts
                        load1_tail = sum(self.demands[n] for n in r1[i+1:])
                        load2_tail = sum(self.demands[n] for n in r2[j+1:])
                        
                        # Check Swapped Capacities
                        if (load1_head + load2_tail > self.capacity) or \
                           (load2_head + load1_tail > self.capacity):
                            continue
                            
                        # Calculate Gain
                        A = r1[i]
                        B = r1[i+1] if i < len(r1)-1 else 0
                        C = r2[j]
                        D = r2[j+1] if j < len(r2)-1 else 0
                        
                        old_cost = self.dist_matrix[A][B] + self.dist_matrix[C][D]
                        new_cost = self.dist_matrix[A][D] + self.dist_matrix[C][B]
                        
                        gain = old_cost - new_cost
                        
                        if gain > best_gain:
                            best_gain = gain
                            move = (r1_idx, r2_idx, i, j)
                            
        if move:
            r1_idx, r2_idx, i, j = move
            r1 = routes[r1_idx]
            r2 = routes[r2_idx]
            
            # Create new routes
            new_r1 = r1[:i+1] + r2[j+1:]
            new_r2 = r2[:j+1] + r1[i+1:]
            
            routes[r1_idx] = new_r1
            routes[r2_idx] = new_r2
            return True
        return False

    # --- Helpers ---
    def _extract_routes(self, tour):
        routes = []
        curr = []
        for node in tour:
            if node == 0:
                if curr:
                    routes.append(curr)
                    curr = []
            else:
                curr.append(node)
        if curr: routes.append(curr)
        return routes

    def _get_route_load(self, route):
        return sum(self.demands[n] for n in route)

    def _calculate_route_cost(self, route):
        cost = 0
        prev = 0
        for node in route:
            cost += self.dist_matrix[prev][node]
            prev = node
        cost += self.dist_matrix[prev][0]
        return cost