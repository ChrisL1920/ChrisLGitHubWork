class Solution(object):
    def searchInsert(self, nums, target):
        if target in nums:
            return nums.index(target)
        else:
            nums.append(target)
            nums.sort()
            return nums.index(target)
              
        
solution = Solution()
result = solution.searchInsert(nums = [1,3,5,6], target = 2)
print(result)