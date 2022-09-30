//
//  Sort.swift
//  LeetCode
//
//  Created by 徐皓东 on 2022/9/28.
//

import Foundation

/// 排序相关练习题
class Sort: BaseCode {
    
    /// 题目链接：[面试题 17.09. 第 k 个数](https://leetcode.cn/problems/get-kth-magic-number-lcci/)
    func getKthMagicNumber(_ k: Int) -> Int {
        var ans = [Int](repeating: 0, count: k)
        ans[0] = 1
        var i3 = 0, i5 = 0, i7 = 0
        for i in 1..<k {
            let a = ans[i3] * 3, b = ans[i5] * 5, c = ans[i7] * 7
            let minV = min(a, b, c)
            if a == minV { i3 += 1 }
            if b == minV { i5 += 1 }
            if c == minV { i7 += 1 }
            ans[i] = minV
        }
        return ans[k - 1]
    }

    /// 题目链接：[1201. 丑数 III](https://leetcode.cn/problems/ugly-number-iii/)
    func nthUglyNumber(_ n: Int) -> Int {
        var ans = [Int](repeating: 0, count: n)
        ans[0] = 1
        var i2 = 0, i3 = 0, i5 = 0
        for i in 1..<n {
            let a = ans[i2] * 2, b = ans[i3] * 3, c = ans[i5] * 5
            let minV = min(a, b, c)
            if a == minV { i2 += 1 }
            if b == minV { i3 += 1 }
            if c == minV { i5 += 1 }
            ans[i] = minV
        }
        return ans[n - 1]
    }
    
//    override var excuteable: Bool { return true }
    
    override func executeTestCode() {
        super.executeTestCode()

    }
}
