//
//  Smapling.swift
//  LeetCode
//
//  Created by haodong xu on 2022/12/12.
//

import Foundation

/// 题目链接：[382. 链表随机节点](https://leetcode.cn/problems/linked-list-random-node/description/)
class RandomListNode {
    var head: ListNode?

    init(_ head: ListNode?) {
        self.head = head
    }

    func getRandom() -> Int {
        var head = head
        var cnt = 0, ans = -1
        while head != nil {
            cnt += 1
            let k = Int.random(in: 1...cnt)
            if k == 1 { ans = head!.val }
            head = head?.next
        }
        return ans
    }
}

class Smapling: BaseCode {

}
