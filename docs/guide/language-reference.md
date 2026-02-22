# Vortex Language Reference

## Types

| Type | Description | Example |
|------|-------------|---------|
| `i64` | 64-bit integer (internally i128) | `42`, `-7` |
| `f64` | 64-bit float | `3.14`, `1e-5` |
| `bool` | Boolean | `true`, `false` |
| `String` | UTF-8 string | `"hello"` |
| `[T]` | Array | `[1, 2, 3]` |
| `(T, U)` | Tuple | `(1, "a")` |
| `Option` | Optional value | `Some(x)`, `None` |
| `Result` | Error handling | `Ok(v)`, `Err(e)` |
| `HashMap` | Key-value map | `hashmap()` |
| Structs | User-defined types | `Point { x: 1.0, y: 2.0 }` |
| Enums | Algebraic data types | `Shape::Circle(5.0)` |

### Constants

```vortex
const PI: f64 = 3.14159
const E: f64 = 2.71828
```

`PI` and `E` are also available as global builtins.

### Type Casting

```vortex
let x = f64(42)       // int to float
let y = i64(3.7)      // float to int (truncates)
let s = to_string(42) // anything to string
let n = parse_int("42")
let f = parse_float("3.14")
```

## Variables

```vortex
let x = 10          // immutable binding
var y = 20          // mutable binding
y = 30              // reassignment (only var)
y += 5              // compound assignment (+=, -=, *=, /=)
```

## Functions

```vortex
fn add(a: i64, b: i64) -> i64 {
    return a + b
}

fn greet(name: String) {
    println(format("Hello, {}!", name))
}
```

### Closures

```vortex
let double = |x: i64| { return x * 2 }
let square = |x| x * x

let nums = [1, 2, 3, 4]
let doubled = map(nums, |x| x * 2)
let evens = filter(nums, |x| x % 2 == 0)
```

### Recursion

```vortex
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}
```

## Control Flow

### If / Else

```vortex
if x > 0 {
    println("positive")
} else if x == 0 {
    println("zero")
} else {
    println("negative")
}
```

### While Loop

```vortex
var i = 0
while i < 10 {
    println(i)
    i = i + 1
}
```

### For Loop

```vortex
for i in range(0, 10) {
    println(i)
}

for item in [1, 2, 3] {
    println(item)
}

// Range syntax
for i in 0..10 {
    println(i)
}
```

### Loop, Break, Continue

```vortex
var count = 0
loop {
    if count >= 5 {
        break
    }
    count = count + 1
    if count == 3 {
        continue
    }
    println(count)
}
```

### Match

```vortex
match value {
    0 => println("zero")
    1 => println("one")
    _ => println("other")
}

match shape {
    Shape::Circle(r) => PI * r * r
    Shape::Rect(w, h) => w * h
}
```

## Structs

```vortex
struct Point {
    x: f64
    y: f64
}

fn distance(p: Point) -> f64 {
    return sqrt(p.x * p.x + p.y * p.y)
}

fn main() {
    let p = Point { x: 3.0, y: 4.0 }
    println(distance(p))    // 5.0
    println(p.x)            // 3.0
}
```

### Methods (impl blocks)

```vortex
impl Point {
    fn distance(self) -> f64 {
        return sqrt(self.x * self.x + self.y * self.y)
    }

    fn translate(self, dx: f64, dy: f64) -> Point {
        return Point { x: self.x + dx, y: self.y + dy }
    }
}
```

### Traits

```vortex
trait Numeric {
    fn zero() -> Self
    fn one() -> Self
    fn add(self: Self, other: Self) -> Self
}
```

## Enums

```vortex
enum Color {
    Red
    Green
    Blue
    Custom(i64, i64, i64)
}

enum Option {
    Some(value)
    None
}

let c = Color::Custom(255, 128, 0)
match c {
    Color::Red => println("red")
    Color::Custom(r, g, b) => println(format("{},{},{}", r, g, b))
    _ => println("other")
}
```

## Operators

### Arithmetic
| Operator | Description |
|----------|-------------|
| `+` `-` `*` `/` | Standard arithmetic |
| `%` | Modulo |
| `**` | Exponentiation |
| `@` | Matrix multiplication |
| `.*` | Elementwise multiplication |
| `./` | Elementwise division |

### Comparison
| Operator | Description |
|----------|-------------|
| `==` `!=` | Equality |
| `<` `>` `<=` `>=` | Ordering |

### Logical
| Operator | Description |
|----------|-------------|
| `&&` | Logical AND |
| `\|\|` | Logical OR |
| `!` | Logical NOT |

### Assignment
| Operator | Description |
|----------|-------------|
| `=` | Assignment |
| `+=` `-=` `*=` `/=` | Compound assignment |

## Comments

```vortex
// This is a line comment
// Vortex uses // for all comments
```

## No Semicolons

Vortex is newline-delimited. No semicolons needed:

```vortex
let x = 1
let y = 2
println(x + y)
```

## String Formatting

```vortex
let name = "Vortex"
let version = 1
println(format("Welcome to {} v{}", name, to_string(version)))
```

## Error Handling

```vortex
let result = ok(42)
if is_ok(result) {
    println(unwrap(result))
}

let maybe = some(10)
let value = unwrap_or(maybe, 0)
```

## Type Aliases

```vortex
type Vec3 = [f64]
type Matrix = [[f64]]
```
