//
//  BitOperation.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/26.
//

import Foundation

/// 位运算相关练习题
class BitOperation: BaseCode {

    /// 题目链接：[面试题 17.19. 消失的两个数字](https://leetcode.cn/problems/missing-two-lcci/)
    func missingTwo(_ nums: [Int]) -> [Int] {
        let n = nums.count + 2
        var missXORSum = 0
        nums.forEach{ missXORSum ^= $0 }
        (1...n).forEach{ missXORSum ^= $0 }
        let diffBit = missXORSum & -missXORSum
        var first = 0
        nums.forEach{ if diffBit & $0 != 0 { first ^= $0 } }
        (1...n).forEach{ if diffBit & $0 != 0 { first ^= $0 }  }
        return [first, missXORSum ^ first]
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
    }
}
