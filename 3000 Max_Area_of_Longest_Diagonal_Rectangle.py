import math as math 
class Solution:
    def areaOfMaxDiagonal(self, dimensions):
        rectangle_dict = {}
        for rectangle_sides in dimensions:
            a = rectangle_sides[0]
            b = rectangle_sides[1]
            area = a * b
            diagonal = a * a + b * b    
            
            if diagonal in rectangle_dict:
                if area > rectangle_dict[diagonal]:
                    rectangle_dict[diagonal] = area

            else:
                rectangle_dict[diagonal] = area

        max_diagonal_rectangle_area = max(rectangle_dict)
        return(rectangle_dict[max_diagonal_rectangle_area])
            
        
solution =  Solution()
result = solution.areaOfMaxDiagonal([[2,5],[7,4],[5,3],[2,4],[3,10],[3,5],[4,5],[4,4],[6,5]])
print(result)

# Diagonal length = sqrt(9 * 9 + 3 * 3) = sqrt(90) ≈ 9.487