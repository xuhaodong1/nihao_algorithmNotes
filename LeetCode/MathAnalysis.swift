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

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
        print(reachNumber(8))
    }
}
