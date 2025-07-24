
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    # For printing the list
    def __str__(self):
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)
    
    def __repr__(self):
        return f"ListNode(val={self.val})"

def build_linked_list(values):
    head = ListNode(values[0])
    current = head

    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next

    return head

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        node_count = 0
        current =  head
        while current is not None:
            node_count += 1
            current = current.next
        # print(node_count)
        middle = int(node_count / 2) 
        current = head 
        for i in range(middle):
            current = current.next
        return current

# Sample input
input_values = [1, 2, 3, 4, 5, 6]
head = build_linked_list(input_values)


# Solve
sol = Solution()
middle = sol.middleNode(head)

# Print from the middle to the end
print("Middle to end of linked list:")
print(middle)