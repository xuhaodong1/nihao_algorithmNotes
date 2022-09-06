//
//  StringType.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/6.
//

import Foundation

/// 字符串相关练习题
class StringType: BaseCode {

    /// 题目链接：[828. 统计子串中的唯一字符](https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/)
    /// 若子串包含重复元素，则重复元素对该子串的不产生贡献
    /// 求每个 s[i] 对答案的贡献, 即求每个 s[i] 可作为多少个子数组的唯一元素。
    func uniqueLetterString(_ s: String) -> Int {
        let n = s.count, chars = [Character](s)
        var map = [Character: Int]()
        var left = [Int](repeating: -1, count: n), right = [Int](repeating: n, count: n)
        var ans = 0
        for i in 0..<n {
            left[i] = i + 1
            if let index = map[chars[i]] {
                left[i] -= (index + 1)
            }
            map[chars[i]] = i
        }
        map.removeAll()
        for i in (0..<n).reversed() {
            right[i] = n - i
            if let index = map[chars[i]] {
                right[i] = index - i
            }
            map[chars[i]] = i
        }
        for i in 0..<n {
            ans += (left[i] * right[i])
        }
        return ans
    }

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
        print(uniqueLetterString("LEETCODE"))
    }
}
