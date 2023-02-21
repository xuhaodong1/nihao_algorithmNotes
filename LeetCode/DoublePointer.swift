//
//  DoublePointer.swift
//  LeetCode
//
//  Created by haodong xu on 2022/10/31.
//

import Foundation

class CustomFunction {
    func f(_ x: Int, _ y: Int) -> Int { return 0 }
}

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

    /// 题目链接：[809. 情感丰富的文字](https://leetcode.cn/problems/expressive-words/)
    func expressiveWords(_ s: String, _ words: [String]) -> Int {
        let n = s.count, sChars = [Character](s)
        var ans = 0
        for word in words {
            let m = word.count, wChars = [Character](word)
            var i = 0, j = 0
            while i < n && j < m {
                if sChars[i] != wChars[j] { break }
                var sCnt = 0, wCnt = 0
                var si = i, wj = j
                while si < n && sChars[si] == sChars[i] {
                    sCnt += 1
                    si += 1
                }
                while wj < m && wChars[wj] == wChars[j] {
                    wCnt += 1
                    wj += 1
                }
                if sCnt != wCnt && (sCnt < 3 || sCnt < wCnt) { break }
                i = si
                j = wj
            }
            if i == n && j == m { ans += 1 }
        }
        return ans
    }

    /// 题目链接：[2565. 最少得分子序列](https://leetcode.cn/problems/subsequence-with-the-minimum-score/description/)
    func minimumScore(_ s: String, _ t: String) -> Int {
        let s = [Character](s), t = [Character](t)
        let n = s.count, m = t.count
        var suf = [Int](repeating: 0, count: n + 1)
        suf[n] = m
        var j = m - 1
        for i in (0..<n).reversed() {
            if j >= 0 && s[i] == t[j] { j -= 1 }
            suf[i] = j + 1
        }
        var ans = suf[0]
        if ans == 0 { return 0 }
        j = 0
        for i in 0..<n {
            if s[i] == t[j] {
                j += 1
                ans = min(ans, suf[i + 1] - j)
            }
        }
        return ans
    }

    /// 题目链接：[1237. 找出给定方程的正整数解](https://leetcode.cn/problems/find-positive-integer-solution-for-a-given-equation/)
    func findSolution(_ customfunction: CustomFunction, _ z: Int) -> [[Int]] {
        var l = 1, r = 1000
        var ans = [[Int]]()
        while l <= 1000 && r > 0 {
            let curr = customfunction.f(l, r)
            if curr == z {
                ans.append([l, r])
                l += 1
                r -= 1
            } else if curr > z {
                r -= 1
            } else {
                l += 1
            }
        }
        return ans
    }

    /// 题目链接：[2570. 合并两个二维数组 - 求和法](https://leetcode.cn/problems/merge-two-2d-arrays-by-summing-values/description/)
    func mergeArrays(_ nums1: [[Int]], _ nums2: [[Int]]) -> [[Int]] {
        let n = nums1.count, m = nums2.count
        var l1 = 0, l2 = 0
        var ans = [[Int]]()
        while l1 < n || l2 < m {
            if l2 == m {
                ans.append(contentsOf: nums1[l1..<n])
                return ans
            } else if l1 == n {
                ans.append(contentsOf: nums2[l2..<m])
                return ans
            } else if nums1[l1][0] == nums2[l2][0] {
                ans.append([nums1[l1][0], nums1[l1][1] + nums2[l2][1]])
                l1 += 1
                l2 += 1
            } else if nums1[l1][0] < nums2[l2][0] {
                ans.append([nums1[l1][0], nums1[l1][1]])
                l1 += 1
            } else {
                ans.append([nums2[l2][0], nums2[l2][1]])
                l2 += 1
            }
        }
        return ans
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(mergeArrays([[2,4],[3,6],[5,5]], [[1,3],[4,3]]))
    }
}
