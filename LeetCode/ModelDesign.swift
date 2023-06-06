//
//  ModelDesign.swift
//  LeetCode
//
//  Created by haodong xu on 2022/2/6.
//

import Foundation

class Bitset {
    private var bits: [Int]
    private var oneCount: Int
    private var zeroCount: Int
    private var isFlip: Bool
    init(_ size: Int) {
        bits = [Int].init(repeating: 0, count: size)
        oneCount = 0
        zeroCount = size
        isFlip = false
    }
    func fix(_ idx: Int) {
        let bit = isFlip ? bits[idx] ^ 1 : bits[idx]
        if bit != 1 {
            oneCount += 1
            zeroCount -= 1
            bits[idx] ^= 1
        }
    }
    func unfix(_ idx: Int) {
        let bit = isFlip ? bits[idx] ^ 1 : bits[idx]
        if bit != 0 {
            zeroCount += 1
            oneCount -= 1
            bits[idx] ^= 1
        }
    }
    func flip() {
        isFlip = !isFlip
        let temp = oneCount
        oneCount = zeroCount
        zeroCount = temp
    }
    func all() -> Bool {
        return oneCount == bits.count
    }
    func one() -> Bool {
        return oneCount > 0
    }
    func count() -> Int {
        return oneCount
    }
    func toString() -> String {
        var ans = ""
        for bit in bits {
            ans += "\(isFlip ? bit ^ 1 : bit)"
        }
        return ans
    }
}

public class ListNode {
    public var val: Int
    public var next: ListNode?
    public var pre: ListNode?
    public init() { self.val = 0; self.next = nil; }
    public init(_ val: Int) { self.val = val; self.next = nil; }
    public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
}

public class TreeNode {
    public var val: Int
    public var left: TreeNode?
    public var right: TreeNode?
    public init() { self.val = 0; self.left = nil; self.right = nil; }
    public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
    public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
        self.val = val
        self.left = left
        self.right = right
    }
}

public class Node {
    public var val: Int
    public var children: [Node]
    public init(_ val: Int) {
        self.val = val
        self.children = []
    }
}

public class SingalNode: NSObject {
    public var val: Int
    public var next: SingalNode?
    public var random: SingalNode?
    public init(_ val: Int) {
        self.val = val
        self.next = nil
        self.random = nil
    }
}

class LRUCache {
    var map: [Int: DListNode]
    var capacity: Int
    var head: DListNode?
    var tail: DListNode?

    init(_ capacity: Int) {
        map = [Int: DListNode]()
        self.capacity = capacity
        head = DListNode(0, 0) // 伪头节点
        tail = DListNode(0, 0) // 伪尾节点
        head?.next = tail
        tail?.pre = head
    }

    func get(_ key: Int) -> Int {
        let node = map[key]
        if node != nil {
            moveNodeToHead(node: node!)
        }
        return node?.val ?? -1
    }

    func put(_ key: Int, _ value: Int) {
        if map.keys.contains(key) {
            let node = map[key]!
            node.val = value
            moveNodeToHead(node: node)
        } else {
            let node = DListNode.init(key, value)
            map[key] = node
            addToHead(node: node)
            if map.count > capacity {
                if let node = removeTail() {
                    map.removeValue(forKey: node.key)
                }
            }
        }
    }

    private func addToHead(node: DListNode) {
        let next = head?.next
        head?.next = node
        node.next = next
        next?.pre = node
        node.pre = head
    }

    private func removeNode(node: DListNode) {
        let preNode = node.pre
        let nextNode = node.next
        preNode?.next = nextNode
        nextNode?.pre = preNode
    }

    private func moveNodeToHead(node: DListNode) {
        removeNode(node: node)
        addToHead(node: node)
    }

    private func removeTail() -> DListNode? {
        let prePreNode = tail?.pre?.pre
        let returnNode = prePreNode?.next
        prePreNode?.next = tail
        tail?.pre = prePreNode
        return returnNode
    }

    public class DListNode {
        public var key: Int
        public var val: Int
        public var next: DListNode?
        public var pre: DListNode?
        public init(_ key: Int, _ val: Int) {
            self.key = key
            self.val = val
        }
    }
}

class AllOne {
    var root: TNode
    var nodes: [String: TNode]
    init() {
        root = TNode.init(0)
        root.next = root
        root.pre = root
        nodes = [String: TNode]()
    }

    func inc(_ key: String) {
        if nodes.keys.contains(key) { // 在链表中
            let curr = nodes[key]!
            let next = curr.next
            if next == root || (next?.cnt ?? 0) > curr.cnt + 1 {
                let new = TNode.init(curr.cnt + 1, key: key)
                curr.insert(node: new)
                nodes[key] = new
            } else {
                next?.keys.insert(key)
                nodes[key] = next
            }
            curr.keys.remove(key)
            if curr.keys.isEmpty {
                curr.remove()
            }

        } else { // 不在链表中
            if root.next == root || root.next!.cnt > 1 {
                let node = TNode.init(1, key: key)
                root.insert(node: node)
                nodes[key] = node
            } else {
                root.next?.keys.insert(key)
                nodes[key] = root.next
            }
        }
    }

    func dec(_ key: String) {
        let curr = nodes[key]!
        if curr.cnt == 1 {
            nodes.removeValue(forKey: key)
        } else {
            let pre = curr.pre
            if pre == root || pre!.cnt < curr.cnt - 1 {
                let node = TNode.init(curr.cnt - 1, key: key)
                pre?.insert(node: node)
                nodes[key] = node
            } else {
                pre?.keys.insert(key)
                nodes[key] = pre
            }
        }
        curr.keys.remove(key)
        if curr.keys.isEmpty {
            curr.remove()
        }
    }

    func getMaxKey() -> String {
        return root.pre == root ? "" : (root.pre?.keys.randomElement() ?? "")
    }

    func getMinKey() -> String {
        return root.next == root ? "" : (root.next?.keys.randomElement() ?? "")
    }


    public class TNode: Equatable {
        public static func == (lhs: AllOne.TNode, rhs: AllOne.TNode) -> Bool {
            return lhs.cnt == rhs.cnt && lhs.keys == rhs.keys
        }

        var pre: TNode?
        var next: TNode?
        var keys: Set<String>
        var cnt: Int
        init(_ cnt: Int) {
            keys = Set<String>()
            self.cnt = cnt
        }

        init(_ cnt: Int, key: String) {
            keys = [key]
            self.cnt = cnt
        }

        func insert(node: TNode?) {
            node?.pre = self
            node?.next = self.next
            self.next = node
            node?.next?.pre = node
        }

        func remove() {
            self.pre?.next = self.next
            self.next?.pre = self.pre
        }
    }
}

class Bank {
    var accounts: [Int]
    let n: Int
    init(_ balance: [Int]) {
        accounts = balance
        n = balance.count
    }

    func transfer(_ account1: Int, _ account2: Int, _ money: Int) -> Bool {
        if account1 <= n && account2 <= n && accounts[account1 - 1] >= money {
            accounts[account1 - 1] -= money
            accounts[account2 - 1] += money
            return true
        }
        return false
    }

    func deposit(_ account: Int, _ money: Int) -> Bool {
        if account <= n {
            accounts[account - 1] += money
            return true
        }
        return false
    }

    func withdraw(_ account: Int, _ money: Int) -> Bool {
        if account <= n && accounts[account - 1] >= money {
            accounts[account - 1] -= money
            return true
        }
        return false
    }
}

class Pair {
    var key: Int
    var value: Int
    var next: Pair?
    var pre: Pair?
    init(key: Int, value: Int) {
        self.key = key
        self.value = value
    }
}

class MyHashMap {
    let base = 769
    var list: [Pair]
    init() {
        list = [Pair]()
        for _ in 0..<base {
            list.append(.init(key: -1, value: -1))
        }
    }

    func put(_ key: Int, _ value: Int) {
        let i = hash(key: key)
        var pair: Pair? = list[i]
        while pair?.key != key && pair != nil {
            pair = pair?.next
        }
        if pair != nil {
            pair?.value = value
        } else {
            var tail: Pair? = list[i]
            while tail?.next != nil {
                tail = tail?.next
            }
            let new = Pair.init(key: key, value: value)
            new.pre = tail
            tail?.next = new
        }
    }

    func get(_ key: Int) -> Int {
        let i = hash(key: key)
        var pair: Pair? = list[i]
        while pair?.key != key && pair != nil {
            pair = pair?.next
        }
        return pair?.value ?? -1
    }

    func remove(_ key: Int) {
        let i = hash(key: key)
        var pair: Pair? = list[i]
        while pair?.key != key && pair != nil {
            pair = pair?.next
        }
        if let pair = pair {
            pair.pre?.next = pair.next
            pair.next = pair.pre
        }
    }

    private func hash(key: Int) -> Int {
        return key % base
    }
}

class Encrypter {
    var mapKeys: [Character: String]
    var mapValues: [String: [Character]]
    var set: [String: Int]
    init(_ keys: [Character], _ values: [String], _ dictionary: [String]) {
        mapKeys = [Character: String]()
        mapValues = [String: [Character]]()
        for i in 0..<keys.count {
            let key = keys[i], value = values[i]
            mapKeys[key] = value
            if mapValues[value] == nil {
                mapValues[value] = [key]
            } else {
                mapValues[value]?.append(key)
            }
        }
        set = [String: Int]()
        for str in dictionary {
            var ans = ""
            for c in str {
                ans += mapKeys[c]!
            }
            set[ans] = (set[ans] ?? 0) + 1
        }
    }

    func encrypt(_ word1: String) -> String {
        return word1.reduce("", { $0 + mapKeys[$1]! })
    }

    func decrypt(_ word2: String) -> Int {
        return set[word2] ?? 0
    }
}

class SegmentTreeNumArray {
    class Node {
        var v: Int = -1
        var l = -1, r = -1
        init() {}
        init(v: Int, l: Int, r: Int) {
            self.v = v
            self.l = l
            self.r = r
        }
        init(l: Int, r: Int) {
            self.l = l
            self.r = r
        }
    }
    var tree: [Node]
    let n: Int
    var nums: [Int]

    init(_ nums: [Int]) {
        n = nums.count
        self.nums = nums
        tree = [Node].init(repeating: Node(), count: n << 2)
        buildTree(cur: 1, l: 1, r: n)
    }

    @discardableResult
    private func buildTree(cur: Int, l: Int, r: Int) -> Node {
        let node = Node(l: l, r: r)
        if l == r {
            node.v = nums[l - 1]
        } else {
            let mid = (l + r) >> 1
            let left = buildTree(cur: cur << 1, l: l, r: mid)
            let right = buildTree(cur: cur << 1 | 1, l: mid + 1, r: r)
            node.v = left.v + right.v
        }
        tree[cur] = node
        return node
    }

    private func update(cur: Int, index: Int, v: Int) {
        if tree[cur].l == index && tree[cur].r == index {
            tree[cur].v = v
            return
        }
        let mid = (tree[cur].l + tree[cur].r) >> 1
        if index <= mid {
            update(cur: cur << 1, index: index, v: v)
        } else {
            update(cur: cur << 1 | 1, index: index, v: v)
        }
        tree[cur].v = tree[cur << 1].v + tree[cur << 1 | 1].v
    }

    private func query(cur: Int, l: Int, r: Int) -> Int {
        if tree[cur].l >= l && tree[cur].r <= r {
            return tree[cur].v
        }
        var sum = 0
        let mid = (tree[cur].l + tree[cur].r) >> 1
        if mid >= l {
            sum += query(cur: cur << 1, l: l, r: r)
        }
        if r > mid {
            sum += query(cur: cur << 1 | 1, l: l, r: r)
        }
        return sum
    }

    func update(_ index: Int, _ val: Int) {
        update(cur: 1, index: index + 1, v: val)
        nums[index] = val
    }

    func sumRange(_ left: Int, _ right: Int) -> Int {
        query(cur: 1, l: left + 1, r: right + 1)
    }
}

class NumArray {
    var tree: [Int]
    var nums: [Int]
    let n: Int
    init(_ nums: [Int]) {
        n = nums.count
        tree = [Int].init(repeating: 0, count: n + 1)
        self.nums = nums
        for i in 0..<n {
            add(i + 1, nums[i])
        }
    }

    func update(_ index: Int, _ val: Int) {
        add(index + 1, val - nums[index])
        nums[index] = val
    }

    func add(_ index: Int, _ val: Int) {
        var index = index
        while index <= n {
            tree[index] += val
            index += lowBit(x: index)
        }
    }

    func query(_ index: Int) -> Int {
        var index = index + 1, ans = 0
        while index > 0 {
            ans += tree[index]
            index -= lowBit(x: index)
        }
        return ans
    }

    func sumRange(_ left: Int, _ right: Int) -> Int {
        return query(right) - query(left - 1)
    }

    private func lowBit(x: Int) -> Int {
        return x & -x
    }
}

class RandomizedSet {
    var set: Set<Int>
    init() {
        set = Set<Int>()
    }

    func insert(_ val: Int) -> Bool {
        if set.contains(val) {
            return false
        }
        set.insert(val)
        return true
    }

    func remove(_ val: Int) -> Bool {
        if !set.contains(val) {
            return false
        }
        set.remove(val)
        return true
    }

    func getRandom() -> Int {
        set.randomElement() ?? -1
    }
}

class NestedInteger {
    var value: Int
    var list = [NestedInteger]()
    init(_ value: Int) {
        self.value = value
    }
    public func isInteger() -> Bool {
        return list.isEmpty
    }
    public func getInteger() -> Int {
        return value
    }
    public func setInteger(value: Int) {
        self.value = value
    }
    public func add(elem: NestedInteger) {
        list.append(elem)
    }
    public func getList() -> [NestedInteger] {
        return list
    }
}

class RecentCounter {
    var queue: [Int]
    init() {
        queue = [Int]()
    }

    func ping(_ t: Int) -> Int {
        let pre = t - 3000
        while !queue.isEmpty && queue.first! < pre { queue.removeFirst() }
        queue.append(t)
        return queue.count
    }
}

class CountIntervals {
    var arr: [[Int]]
    init() {
        arr = [[Int]]()
    }

    func add(_ left: Int, _ right: Int) {
        if arr.isEmpty {
            arr.append([left, right])
            return
        }
        var l = 0, r = arr.count
        while l < r {
            let mid = (l + r) >> 1
            if arr[mid][0] > left {
                r = mid
            } else {
                l = mid + 1
            }
        }
        arr.insert([left, right], at: l)
    }

    func count() -> Int {
        if arr.isEmpty {
            return 0
        }
        var curr = arr[0][1] + 1
        var ans = arr[0][1] - arr[0][0] + 1
        for i in 1..<arr.count {
            if arr[i][1] < curr {
                continue
            }
            if arr[i][0] >= curr {
                ans += arr[i][1] - arr[i][0] + 1
            } else {
                ans += arr[i][1] - curr + 1
            }
            curr = arr[i][1] + 1
        }
        return ans
    }

    func inorderSuccessor(_ root: TreeNode?, _ p: TreeNode?) -> TreeNode? {
        guard let root = root, let p = p else {
            return nil
        }
        var queue: [TreeNode] = [root], curr: TreeNode? = root
        var ans: TreeNode?
        while !queue.isEmpty || curr != nil {
            while curr?.left != nil {
                queue.append(curr!.left!)
                curr = curr?.left!
            }
            let node = queue.removeLast()
            if node.val < p.val {
                ans = node
                break
            }
            curr = curr?.right
        }
        return ans
    }
}

class MinStack {
    var stack: [Int]
    var nums: [Int]

    init() {
        stack = [Int]()
        nums = [Int]()
    }

    func push(_ val: Int) {
        if stack.isEmpty {
            stack.append(val)
        } else {
            stack.append(stack.last! < val ? stack.last! : val)
        }
        nums.append(val)
    }

    func pop() {
        stack.removeLast()
        nums.removeLast()
    }

    func top() -> Int {
        return nums.last!
    }

    func getMin() -> Int {
        return stack.last!
    }
}

class Solution2 {
    var radius: Double
    var xCenter: Double
    var yCenter: Double
    init(_ radius: Double, _ x_center: Double, _ y_center: Double) {
        self.radius = radius
        self.xCenter = x_center
        self.yCenter = y_center
    }

    func randPoint() -> [Double] {
        var x: Double = 0.0, y: Double = 0.0
        repeat {
            x = Double.random(in: -radius...radius)
            y = Double.random(in: -radius...radius)
        } while x + y > radius
        return [x + xCenter, y + yCenter]
    }
}

class TextEditor {
    var preText: [Character]
    var tailText: [Character]
    init() {
        preText = [Character]()
        tailText = [Character]()
    }

    func addText(_ text: String) {
        for char in text {
            preText.append(char)
        }
    }

    func deleteText(_ k: Int) -> Int {
        let cnt = preText.count
        if k > cnt {
            preText.removeAll()
            return cnt
        } else {
            preText.removeSubrange(cnt-k..<cnt)
            return k
        }
    }

    func cursorLeft(_ k: Int) -> String {
        for _ in 0..<k {
            if preText.count > 0 {
                tailText.append(preText.removeLast())
            } else {
                break
            }
        }
        var ans = ""
        let n = preText.count
        for i in (0..<min(n, 10)).reversed() {
            ans.append(preText[n - i - 1])
        }
        return ans
    }

    func cursorRight(_ k: Int) -> String {
        for _ in 0..<k {
            if tailText.count > 0 {
                preText.append(tailText.removeLast())
            } else {
                break
            }
        }
        var ans = ""
        let n = preText.count
        for i in (0..<min(n, 10)).reversed() {
            ans.append(preText[n - i - 1])
        }
        return ans
    }
}

class MyCalendarThree {
    var map: [Int: Int]
    init() {
        map = [Int: Int]()
    }

    func book(_ start: Int, _ end: Int) -> Int {
        var ans = 0
        var maxBook = 0
        map[start] = (map[start] ?? 0) + 1
        map[end] = (map[end] ?? 0) - 1
        for item in map.sorted(by: { item1, item2 in
            return item1.key < item2.key
        }) {
            let freq = item.value
            maxBook += freq
            ans = max(maxBook, ans)
        }
        return ans
    }
}

class Solution3 {
    var rects: [[Int]]
    var preSum: [Int]

    init(_ rects: [[Int]]) {
        self.rects = rects
        let n = rects.count
        preSum = [Int].init(repeating: 0, count: n + 1)
        for i in 1...n {
            //            preSum[i] = preSum[i - 1] + (rects[i - 1][2] - rects[i - 1][0] + 1)(rects[i - 1][3] - rects[i - 1][1] + 1)
        }
    }

    func pick() -> [Int] {
        let select = Int.random(in: 1...preSum.max()!)
        var l = 1, r = rects.count
        while l < r {
            let mid = l + (r - l) >> 1
            if preSum[mid] + 1 >= select {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return [Int.random(in: rects[l - 1][0]...rects[l - 1][2]), Int.random(in: rects[l - 1][1]...rects[l - 1][3])]
    }
}


//class Solution4 {
//
//    init(_ n: Int, _ blacklist: [Int]) {
//
//    }
//
//    func pick() -> Int {
//
//    }
//}

class Codec {
    var longMap = [String: String](), shortMap = [String: String]()
    var id = 0
    let pre = "http://tinyUrl.com/"
    // Encodes a URL to a shortened URL.
    func encode(_ longUrl: String) -> String {
        if let shortUrl = longMap[longUrl]  {
            return shortUrl
        }
        id += 1
        let shortUrl = pre + "\(id)"
        longMap[longUrl] = shortUrl
        shortMap[shortUrl] = longUrl
        return shortUrl
    }

    // Decodes a shortened URL to its original URL.
    func decode(_ shortUrl: String) -> String {
        return shortMap[shortUrl] ?? ""
    }
}

class MyCalendar {
    var list = [(Int, Int)]()
    init() {}

    func book(_ start: Int, _ end: Int) -> Bool {
        if list.contains(where: { (s, e) in
            s < end && start < e
        }) {
            list.append((start, end))
            return false
        }
        list.append((start, end))
        return false
    }
}

class SmallestInfiniteSet {
    var removeList = Set<Int>()
    var addList = Set<Int>()
    var curr = 0
    init() {}

    func popSmallest() -> Int {
        if let num = addList.min(), num < curr + 1 {
            addList.remove(num)
            removeList.insert(num)
            return num
        }
        curr += 1
        removeList.insert(curr)
        return curr
    }

    func addBack(_ num: Int) {
        if removeList.contains(num) {
            addList.insert(num)
            removeList.remove(num)
        }
    }
}

class MagicDictionary {
    private var arr: [String]
    init() {
        arr = [String]()
    }

    func buildDict(_ dictionary: [String]) {
        arr = dictionary
    }

    func search(_ searchWord: String) -> Bool {
        let n = searchWord.count
        for str in arr {
            if str.count != n { continue }
            var diff = 0
            var strIndex = str.startIndex
            for char in searchWord {
                if char != str[strIndex] {
                    diff += 1
                }
                if diff >= 2 {
                    break
                }
                strIndex = str.index(after: strIndex)
            }
            if diff == 1 {
                return true
            }
        }
        return false
    }
}

class WordFilter {
    let aV = Int(Character.init("a").asciiValue!)
    var head = WordNode(), tail = WordNode()

    init(_ words: [String]) {
        for (wi, word) in words.enumerated() {
            let n = word.count
            let chars = [Character](word)
            var hN = head, tN = tail
            for i in 0..<n {
                var cV = Int(chars[i].asciiValue!)
                if hN.children[cV - aV] == nil {
                    hN.children[cV - aV] = WordNode()
                }
                hN.children[cV - aV]?.indices.append(wi)
                hN = hN.children[cV - aV]!

                cV = Int(chars[n - i - 1].asciiValue!)
                if nil == tN.children[cV - aV] {
                    tN.children[cV - aV] = WordNode()
                }
                tN.children[cV - aV]?.indices.append(wi)
                tN = tN.children[cV - aV]!
            }
        }
    }

    func query(root: WordNode, str: [Character]) -> [Int]? {
        var ans: WordNode?
        var node = root
        for char in str {
            let cV = Int(char.asciiValue!)
            if let _ = node.children[cV - aV] {
                node = node.children[cV - aV]!
                ans = node
            } else {
                return nil
            }
        }
        return ans?.indices
    }

    func f(_ pref: String, _ suff: String) -> Int {
        if let l1 = query(root: head, str: [Character](pref)), let l2 = query(root: tail, str: suff.reversed()) {
            var i = l1.count - 1, j = l2.count - 1
            while i >= 0 && j >= 0 {
                if l1[i] == l2[j] {
                    return l1[i]
                } else if l1[i] > l2[j] {
                    i -= 1
                } else {
                    j -= 1
                }
            }
        }
        return -1
    }

    class WordNode {
        var children: [WordNode?]
        var indices: [Int]
        init() {
            children = [WordNode?].init(repeating: nil, count: 26)
            indices = [Int]()
        }
    }
}

public class FNode {
    public var val: Bool
    public var isLeaf: Bool
    public var topLeft: FNode?
    public var topRight: FNode?
    public var bottomLeft: FNode?
    public var bottomRight: FNode?
    public init(_ val: Bool, _ isLeaf: Bool) {
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = nil
        self.topRight = nil
        self.bottomLeft = nil
        self.bottomRight = nil
    }
}

class MovingAverage {
    private let size: Int
    private var arr: [Int]
    private var l = 0, r = 0
    private var sum = 0

    init(_ size: Int) {
        self.size = size
        self.arr = [Int]()
    }

    func next(_ val: Int) -> Double {
        arr.append(val)
        sum += val
        r += 1
        if r - l > size {
            sum -= arr[l]
            l += 1
        }
        return Double(sum) / Double(r - l)
    }
}


class MyCalendarTwo {
    var list = [(Int, Int)]()

    init() {}

    func book(_ start: Int, _ end: Int) -> Bool {
        let n = list.count
        if n < 2 {
            list.append((start, end))
            return true
        }
        for i in 0..<n-1 {
            if let a1 = getCommon(s1: list[i], s2: (start, end)) {
                for j in i+1..<n {
                    if let a2 = getCommon(s1: list[j], s2: (start, end)),
                       let _ = getCommon(s1: a1, s2: a2) {
                        return false
                    }
                }
            }
        }
        func getCommon(s1: (Int, Int), s2: (Int, Int)) -> (Int, Int)? {
            if s1.1 <= s2.0 || s2.1 <= s1.0 {
                return nil
            }
            return (max(s1.0, s2.0), min(s1.1, s2.1))
        }
        list.append((start, end))
        return true
    }
}


class FoodRatings {
    var fs = [String : (String, Int)]()
    var cs = [String : [Int: Set<String>]]()
    init(_ foods: [String], _ cuisines: [String], _ ratings: [Int]) {
        let n = foods.count
        for i in 0..<n {
            fs[foods[i]] = (cuisines[i], ratings[i])
            if let rs = cs[cuisines[i]] {
                if let _ = rs[ratings[i]] {
                    cs[cuisines[i]]![ratings[i]]!.insert(foods[i])
                } else {
                    cs[cuisines[i]]![ratings[i]] = [foods[i]]
                }
            } else {
                cs[cuisines[i]] = [ratings[i] : [foods[i]]]
            }
        }
    }

    func changeRating(_ food: String, _ newRating: Int) {
        let cuisine = self.fs[food]!.0
        self.cs[cuisine]![self.fs[food]!.1]!.remove(food)
        if self.cs[cuisine]![self.fs[food]!.1]!.count == 0 {
            self.cs[cuisine]![self.fs[food]!.1] = nil
        }
        if let _ = self.cs[cuisine]![newRating] {
            self.cs[cuisine]![newRating]!.insert(food)
        } else {
            self.cs[cuisine]![newRating] = [food]
        }
        self.fs[food]!.1 = newRating
    }

    func highestRated(_ cuisine: String) -> String {
        let dict = cs[cuisine]!.max { item1, item2 in
            return item1.key < item2.key
        }
        return dict?.value.min() ?? ""
    }
}

class CBTInserter {
    var candidate = [TreeNode]()
    var root: TreeNode!
    init(_ root: TreeNode?) {
        guard let root = root else { return }
        self.root = root
        var queue: [TreeNode] = [root]
        while !queue.isEmpty {
            let curr = queue.removeFirst()
            if let left = curr.left {
                queue.append(left)
            }
            if let right = curr.right {
                queue.append(right)
            }
            if curr.left == nil || curr.right == nil {
                candidate.append(curr)
            }
        }
    }

    func insert(_ val: Int) -> Int {
        let node = TreeNode(val)
        let res = candidate.first!
        if res.left == nil {
            res.left = node
        } else {
            res.right = node
            candidate.removeFirst()
        }
        candidate.append(node)
        return res.val
    }

    func get_root() -> TreeNode? {
        return root
    }
}

class MyCircularQueue {
    var queue: [Int]
    var head: Int
    var tail: Int
    var n: Int
    var size: Int

    init(_ k: Int) {
        queue = [Int](repeating: -1, count: k)
        head = 0
        tail = 0
        n = k
        size = 0
    }

    func enQueue(_ value: Int) -> Bool {
        if isFull() {
            return false
        }
        queue[tail] = value
        tail = (tail + 1) % n
        size += 1
        return true
    }

    func deQueue() -> Bool {
        if isEmpty() {
            return false
        }
        queue[head] = -1
        head = (head + 1) % n
        size -= 1
        return true
    }

    func Front() -> Int {
        if isEmpty() {
            return -1
        }
        return queue[head]
    }

    func Rear() -> Int {
        if isEmpty() {
            return -1
        }
        return queue[(tail - 1 + n) % n]
    }

    func isEmpty() -> Bool {
        return head == tail && size == 0
    }

    func isFull() -> Bool {
        return head == tail && size == n
    }
}

class DisorganizeArray {
    private var originNums: [Int]
    private let n: Int
    init(_ nums: [Int]) {
        originNums = nums
        n = nums.count
    }

    func reset() -> [Int] {
        return originNums
    }

    func shuffle() -> [Int] {
        var ans = originNums
        for i in 0..<n {
            ans.swapAt(i, Int.random(in: i..<n))
        }
        return ans
    }
}

class OrderedStream {
    var order: [String]
    let n: Int
    var ptr: Int

    init(_ n: Int) {
        self.n = n
        ptr = 0
        order = [String].init(repeating: "", count: n)
    }

    func insert(_ idKey: Int, _ value: String) -> [String] {
        guard idKey > 0 && idKey <= n else { return [] }
        order[idKey - 1] = value
        var ans = [String]()
        while ptr < n && order[ptr] != "" {
            ans.append(order[ptr])
            ptr += 1
        }
        return ans
    }
}

class MyCircularDeque {
    var deque: [Int]
    var head: Int
    var rear: Int
    var k: Int

    init(_ k: Int) {
        head = 0
        rear = 0
        self.k = k
        deque = [Int](repeating: -1, count: k + 1)
    }

    func insertFront(_ value: Int) -> Bool {
        if isFull() {
            return false
        }
        deque[head] = value
        head = (head + k) % (k + 1)
        return true
    }

    func insertLast(_ value: Int) -> Bool {
        if isFull() {
            return false
        }
        rear = (rear + 1) % (k + 1)
        deque[rear] = value
        return true
    }

    func deleteFront() -> Bool {
        if isEmpty() {
            return false
        }
        deque[head] = -1
        head = (head + 1) % (k + 1)
        return true
    }

    func deleteLast() -> Bool {
        if isEmpty() {
            return false
        }
        deque[rear] = -1
        rear = (rear + k) % (k + 1)
        return true
    }

    func getFront() -> Int {
        if isEmpty() {
            return -1
        }
        return deque[(head + 1) % (k + 1)]
    }

    func getRear() -> Int {
        if isEmpty() {
            return -1
        }
        return deque[rear]
    }

    func isEmpty() -> Bool {
        return rear == head
    }

    func isFull() -> Bool {
        return (head - rear) == 1
    }
}

/// 题目链接：[707. 设计链表](https://leetcode.cn/problems/design-linked-list/)
class MyLinkedList {
    class SingalNode {
        var val: TreeNode
        var next: SingalNode?
        init(_ val: TreeNode) {
            self.val = val
            self.next = nil
        }
    }

    var root: SingalNode?
    var cnt: Int

    init() {
        root = SingalNode(TreeNode.init(-1))
        cnt = 0
    }

    func addAtTail(_ val: TreeNode) {
        addAtIndex(cnt, val)
    }

    func addAtIndex(_ index: Int, _ val: TreeNode) {
        if index > cnt { return }
        var index = index
        if index < 0 { index = 0 }
        var node = root
        for _ in 0..<index {
            node = node?.next
        }
        let next = node?.next
        node?.next = SingalNode(val)
        node?.next?.next = next
        cnt += 1
    }

    func deleteAtIndex(_ index: Int) -> TreeNode? {
        if index >= cnt || index < 0 { return nil }
        var node = root
        for _ in 0..<index {
            node = node?.next
        }
        let ans = node?.next
        node?.next = node?.next?.next
        cnt -= 1
        return ans?.val
    }
}

class StreamChecker {
    var root = Trie()
    var mx = 0
    let charA = Character("a")
    var search = [Character]()

    init(_ words: [String]) {
        for word in words {
            insert(word)
        }
    }

    private func insert(_ word: String) {
        mx = max(word.count, mx)
        var curr = root
        for char in word.reversed() {
            let index = Int(char.asciiValue! - charA.asciiValue!)
            if curr.children[index] == nil {
                curr.children[index] = Trie()
            }
            curr = curr.children[index]!
        }
        curr.isEnd = true
    }

    func query(_ letter: Character) -> Bool {
        search.insert(letter, at: 0)
        if search.count > mx {
            search.removeLast()
        }
        var curr = root
        for i in 0..<min(mx, search.count) {
            let index = Int(search[i].asciiValue! - charA.asciiValue!)
            if curr.children[index] == nil { return false }
            else if curr.children[index]?.isEnd == true { return true }
            curr = curr.children[index]!
        }
        return false
    }

    class Trie {
        var children = [Trie?](repeating: nil, count: 26)
        var isEnd = false
    }
}
