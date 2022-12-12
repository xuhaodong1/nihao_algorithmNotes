//
//  Smapling.swift
//  LeetCode
//
//  Created by haodong xu on 2022/12/12.
//

import Foundation

/// 蓄水抽样相关练习题

/// 题目链接：[382. 链表随机节点](https://leetcode.cn/problems/linked-list-random-node/description/)
class RandomListNode {
    private var head: ListNode?

    init(_ head: ListNode?) {
        self.head = head
    }

    func getRandom() -> Int {
        var head = head
        var cnt = 1, ans = -1
        while head != nil {
            if Int.random(in: 1...cnt) == 1 { ans = head!.val }
            head = head?.next
            cnt += 1
        }
        return ans
    }
}

/// 题目链接：[398. 随机数索引](https://leetcode.cn/problems/random-pick-index/description/)
class RandomNumIndex {
    private var nums: [Int]

    init(_ nums: [Int]) {
        self.nums = nums
    }

    func pick(_ target: Int) -> Int {
        let n = nums.count
        var ans = -1, cnt = 0
        for (i, num) in nums.enumerated() where num == target {
            if Int.random(in: 0...cnt) == 0 { ans = i }
            cnt += 1
        }
        return ans
    }
}

class Smapling: BaseCode {

}
