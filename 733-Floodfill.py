class Solution(object):
    def floodFill(self, image, sr, sc, color):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type color: int
        :rtype: List[List[int]]
        """
        # for i in image:
        #     print(i)

        # print(image[sr][sc])
        
        # print(len(image))
        # print(len(image[0]))
        start_color = (image[sr][sc])
        def paintitselfandneighbors(row, column):
            print('trying to paint', row, column)
            # check if current px exists
            if row < 0 or column < 0 or row >= len(image) or column >= len(image[0]) or image[row][column] == color or image[row][column] != start_color:
                return 
            
            # check if current px shares same color as starting px
            # paint current pixel 
            image[row][column] = color

            # paint top pixel and its neighbors
            paintitselfandneighbors(row - 1, column)

            # paint left
            paintitselfandneighbors(row, column - 1)


            # paint bottom
            paintitselfandneighbors(row + 1, column)

            # paint right
            paintitselfandneighbors(row, column + 1)

        paintitselfandneighbors(sr, sc)
        for i in image:
            print(i)
        return image 
solution =  Solution()
result = solution.floodFill(image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2)
print(result)