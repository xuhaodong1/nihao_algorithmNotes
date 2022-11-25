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
}
