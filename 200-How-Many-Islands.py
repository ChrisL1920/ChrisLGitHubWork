class Solution(object):
    def numIslands(self, grid):
        num_islands = 0
        visited = {(0, 0)}

        def traverse_island(start_node):
            queue = [start_node]
            while len(queue) != 0:
                node = queue.pop(0)
                row, col = node
                # print(row, col)
                if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
                    continue
                if grid[row][col] != '1':
                    continue
                if node in visited:
                    continue
                queue.append((row - 1, col))
                queue.append((row , col - 1))
                queue.append((row + 1, col))
                queue.append((row , col + 1))
                visited.add(node)


        for row_idx, row in enumerate(grid):
            for col_idx, col in enumerate(row):
                coord = (row_idx, col_idx)
                if col == "1" and coord not in visited:
                    traverse_island((coord))
                    num_islands += 1 

        return num_islands

solution =  Solution()
result = solution.numIslands([["1"]])
print(result)
    