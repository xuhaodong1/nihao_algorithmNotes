//
//  Stack.swift
//  LeetCode
//
//  Created by haodong xu on 2022/11/6.
//

import Foundation

/// 题目链接：[895. 最大频率栈](https://leetcode.cn/problems/maximum-frequency-stack/description/)
class FreqStack {
    var cnts = [Int: Int]() // 数字的出现频率
    var stacks = [[Int]]() // 以出现次数相同的组成的序列, stacks[i] 表示出现次数为 i + 1 组成的序列
    init() { }
    func push(_ val: Int) {
        var cnt = cnts[val, default: 0]
        if cnt == stacks.count { stacks.append([]) } // 有一个新的cnt
        stacks[cnt].append(val)
        cnts[val, default: 0] += 1
    }

    func pop() -> Int {
        let back = stacks.count - 1
        let val = stacks[back].removeLast()
        if stacks[back].isEmpty { stacks.removeLast() }
        cnts[val, default: 0] -= 1
        return val
    }
}


/// 栈等相关练习题
class Stack: BaseCode {

    /// 题目链接：[1106. 解析布尔表达式](https://leetcode.cn/problems/parsing-a-boolean-expression/description/)
    func parseBoolExpr(_ expression: String) -> Bool {
        let eChars = [Character](expression)
        var stack = [String]()
        for char in eChars {
            if char == "!" || char == "&" || char == "|" || char == "t" || char == "f" {
                stack.append("\(char)")
            } else if char == ")" {
                var arr = [String]()
                while stack.last != "!" && stack.last != "&" && stack.last != "|" { arr.append(stack.removeLast()) }
                let last = stack.removeLast()
                if last == "!" { stack.append(arr.first! == "t" ? "f" : "t") }
                else if last == "&" { stack.append(arr.contains("f") ? "f" : "t") }
                else if last == "|" { stack.append(arr.contains("t") ? "t" : "f") }
            }
        }
        return stack.first == "t"
    }
}
