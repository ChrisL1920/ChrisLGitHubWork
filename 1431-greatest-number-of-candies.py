class Solution(object):
    def kidsWithCandies(self, candies, extraCandies):
        # result = list(map(lambda i: i + extraCandies >= max(candies), candies))
        result = [i + extraCandies >= max(candies) for i in candies]

        return result

solution =  Solution()
result = solution.kidsWithCandies([4,2,1,1,2], 1)
print(result)


