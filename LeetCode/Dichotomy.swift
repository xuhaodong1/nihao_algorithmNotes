//
//  Dichotomy.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/12.
//

import Foundation

/// 二分相关练习题
class Dichotomy: BaseCode {

    /// 题目链接：[1608. 特殊数组的特征值](https://leetcode.cn/problems/special-array-with-x-elements-greater-than-or-equal-x/)
    func specialArray(_ nums: [Int]) -> Int {
        let n = nums.count, nums = nums.sorted()
        var l = 0, r = Int.max
        while l < r {
            let mid = l + (r - l) >> 1
            if mid >= getCnt(x: mid) {
                r = mid
            } else {
                l = mid + 1
            }
        }
        func getCnt(x: Int) -> Int {
            var l = 0, r = n - 1
            while l < r {
                // mid = (l + r) >> 1 每次找到左边, 因此需要构造 l = mid + 1
                // 如 l = 0, r = 1, 则 mid = 0, 执行 if 结果为 l = mid, 则会造成死循环
                let mid = (l + r) >> 1
                if nums[mid] >= x {
                    r = mid
                } else {
                    l = mid + 1
                }
            }
            return nums[l] >= x ? n - l : 0 // 边界判断
        }
        return getCnt(x: l) == l ? l : -1
    }

    /// 题目链接：[1201. 丑数 III](https://leetcode.cn/problems/ugly-number-iii/)
    func nthUglyNumber(_ n: Int, _ a: Int, _ b: Int, _ c: Int) -> Int {
        func gcd(a: Int, b: Int) -> Int {
            return b == 0 ? a : gcd(a: b, b: a % b)
        }
        func lcm(a: Int, b: Int) -> Int {
            return a * b / gcd(a: a, b: b)
        }
        func getCnt(num: Int) -> Int { // 容斥原理
            return num / a + num / b + num / c - num / lcmAB - num / lcmAC - num / lcmBC + num / lcmABC
        }
        let lcmAB = lcm(a: a, b: b), lcmAC = lcm(a: a, b: c), lcmBC = lcm(a: b, b: c), lcmABC = lcm(a: a, b: lcmBC)
        var l = 1, r = min(a, b, c) * n
        while l < r {
            let mid = (l + r) >> 1
            if getCnt(num: mid) < n {
                l = mid + 1
            } else {
                r = mid
            }
        }
        return l
    }

    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(nthUglyNumber(5, 2, 3, 3))
    }

}
