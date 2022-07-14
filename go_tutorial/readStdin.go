package main

import (
    "fmt"
)

func readStdin() {
    var input string
    fmt.Scanln(&input)
    fmt.Println(input)
}

func main() {
    readStdin()
}
