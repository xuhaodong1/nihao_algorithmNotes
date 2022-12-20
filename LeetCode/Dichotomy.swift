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

    /// 题目链接：[792. 匹配子序列的单词数](https://leetcode.cn/problems/number-of-matching-subsequences/description/)
    func numMatchingSubseq(_ s: String, _ words: [String]) -> Int {
        let a = Character("a").asciiValue!
        var ans = 0
        var positions = [[Int]](repeating: [], count: 26)
        for (i, char) in s.enumerated() {
            positions[Int(char.asciiValue! - a)].append(i)
        }
        for word in words {
            var isOk = true, idx = -1
            for (_, char) in word.enumerated() where isOk {
                let arr = positions[Int(char.asciiValue! - a)]
                var l = 0, r = arr.count - 1
                while l < r { // 找到第一个比idx下标大的
                    let mid = (l + r) >> 1
                    if arr[mid] > idx { r = mid }
                    else { l = mid + 1 }
                }
                if r < 0 || arr[r] <= idx { isOk = false }
                else { idx = arr[r] }
            }
            if isOk { ans += 1 }
        }
        return ans
    }
    
    /// 题目链接：[878. 第 N 个神奇数字](https://leetcode.cn/problems/nth-magical-number/description/)
    func nthMagicalNumber(_ n: Int, _ a: Int, _ b: Int) -> Int {
        let lcm = a * b / gcd(a, b), MOD = Int(1e9 + 7)
        var left = 1, right = n * max(a, b)
        while left < right {
            let mid = left + (right - left) >> 1
            if mid / a + mid / b - mid / lcm >= n { right = mid }
            else { left = mid + 1 }
        }
        func gcd(_ a: Int, _ b: Int) -> Int {
            return b == 0 ? a : gcd(b, a % b)
        }
        return right % MOD
    }

    /// 题目链接：[1760. 袋子里最少数目的球](https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/description/)
    func minimumSize(_ nums: [Int], _ maxOperations: Int) -> Int {
        var left = 1, right = nums.max()!
        while left < right {
            let mid = (left + right) >> 1
            let ops = nums.reduce(0) { $0 + (($1 - 1) / mid) }
            if ops > maxOperations {
                left = mid + 1
            } else {
                right = mid
            }
        }
        return left
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(minimumSize([1], 1))
    }

}
