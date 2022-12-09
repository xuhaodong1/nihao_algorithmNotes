//
//  MathAnalysis.swift
//  LeetCode
//
//  Created by haodong xu on 2022/11/4.
//

import Foundation

/// 数学分析等相关练习题
class MathAnalysis: BaseCode {

    /// 题目链接：[754. 到达终点数字](https://leetcode.cn/problems/reach-a-number/description/)
    /// 参考链接：https://leetcode.cn/problems/reach-a-number/solutions/1947254/fen-lei-tao-lun-xiang-xi-zheng-ming-jian-sqj2/
    func reachNumber(_ target: Int) -> Int {
        let target = abs(target)
        var s = 0, i = 1
        while s < target || (s - target) % 2 == 1 {
            s += i
            i += 1
        }
        return i - 1
    }

    /// 题目链接：[775. 全局倒置与局部倒置](https://leetcode.cn/problems/global-and-local-inversions/description/)
    func isIdealPermutation(_ nums: [Int]) -> Bool {
        return zip((0..<nums.count), nums).allSatisfy { abs($1 - $0) <= 1 }
    }

    /// 题目链接：[891. 子序列宽度之和](https://leetcode.cn/problems/sum-of-subsequence-widths/description/)
    /// 类似题目：
    /// [828. 统计子串中的唯一字符](https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/)
    /// [907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)
    /// [2104. 子数组范围和](https://leetcode.cn/problems/sum-of-subarray-ranges/)
    func sumSubseqWidths(_ nums: [Int]) -> Int {
        let nums = nums.sorted(), n = nums.count, MOD = Int(1e9 + 7)
        var ans = 0, pow2s = [Int](repeating: 0, count: n)
        pow2s[0] = 1
        for i in 1..<n {
            pow2s[i] = (pow2s[i - 1] * 2) % MOD
        }
        for (i, num) in nums.enumerated() {
            ans += ((pow2s[i] - pow2s[n - 1 - i]) * num)
        }
        return ((ans % MOD) + MOD) % MOD
    }

    /// 题目链接：[50. Pow(x, n)](https://leetcode.cn/problems/powx-n/description/)
    func myPow(_ x: Double, _ n: Int) -> Double {
        func quickMul(_ x: Double, _ n: Int) -> Double {
            var ans = 1.0, x_contribute = x, n = n
            while n > 0 {
                if n & 1 == 1 { ans *= x_contribute }
                x_contribute *= x_contribute
                n /= 2
            }
            return ans
        }
        return n > 0 ? quickMul(x, n) : 1.0 / quickMul(x, n)
    }

    /// 题目链接：[795. 区间子数组个数](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/description/)
    func numSubarrayBoundedMax(_ nums: [Int], _ left: Int, _ right: Int) -> Int {
        let n = nums.count
        var outIndex = -1, containIndex = -1
        var ans = 0
        for i in 0..<n {
            if nums[i] > right { outIndex = i }
            if nums[i] >= left { containIndex = i }
            ans += containIndex - outIndex
        }
        return ans
    }

    /// 题目链接：[1780. 判断一个数字是否可以表示成三的幂的和](https://leetcode.cn/problems/check-if-number-is-a-sum-of-powers-of-three/description/)
    func checkPowersOfThreeByMath(_ n: Int) -> Bool {
        let maxPower = 15
        var powers3 = [Int](repeating: 1, count: maxPower), n = n
        for i in 1..<maxPower { powers3[i] = powers3[i - 1] * 3 }
        for i in (0..<maxPower).reversed() where n > 0 && powers3[i] <= n { n -= powers3[i] }
        return n == 0
    }

    /// 题目链接：[1780. 判断一个数字是否可以表示成三的幂的和](https://leetcode.cn/problems/check-if-number-is-a-sum-of-powers-of-three/description/)
    func checkPowersOfThreeBySystem(_ n: Int) -> Bool {
        var n = n
        while n != 0 {
            if n % 3 == 2 { return false }
            n /= 3
        }
        return true
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(myPow(2, 4))
    }
}
