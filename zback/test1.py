from typing import List


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res = self.reversePrint(head.next) + [head.val] if head else []
        return res


if __name__ == '__main__':
    head = ListNode(1)
    head_next = ListNode(3)
    head_next_next = ListNode(2)
    head.next = head_next
    head.next.next = head_next_next

    sol = Solution()
    print(sol.reversePrint(head))
