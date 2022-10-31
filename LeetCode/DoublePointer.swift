//
//  DoublePointer.swift
//  LeetCode
//
//  Created by haodong xu on 2022/10/31.
//

import Foundation

/// 双指针相关练习题
class DoublePointer: BaseCode {

    /// 题目链接：[481. 神奇字符串](https://leetcode.cn/problems/magical-string/)
    func magicalString(_ n: Int) -> Int {
        var cnt = 1
        var i = 2, v = 1
        var curr = [1, 2, 2]
        while curr.count < n {
            curr.append(contentsOf: [Int](repeating: v, count: curr[i]))
            if v == 1 { cnt += curr[i] }
            i += 1
            v ^= 3
        }
        if curr.count > n && curr.last! == 1 { cnt -= 1 }
        return cnt
    }

}
