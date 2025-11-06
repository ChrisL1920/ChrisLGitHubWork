from typing import Optional, List


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
        #For printing the list
    def __str__(self):
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)
    
    def __repr__(self):
        return f"ListNode(val={self.val})"


class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        
        for _ in range(k):
            node = head
            count = 1
            while node.next is not None:
                node = node.next
                count += 1
            node.next = head 
            s = head
            for _ in range(count - 2):
                s = s.next
            s.next = None 
            head = node
        print(head)
        


input = [1, 2, 3, 4, 5]

def returnHead(aList: List) -> ListNode:
    head = ListNode(aList[0])
    current = head 
    for i in aList[1:]:
        node = ListNode(i)
        current.next = node
        current = node
    return head


result1 = returnHead(input)

solution =  Solution()
result = solution.rotateRight(result1, 3)
print(result)
    
