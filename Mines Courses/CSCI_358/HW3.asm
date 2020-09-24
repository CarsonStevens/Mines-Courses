#Carson Stevens
#2/8/2017
#Decription: To write a program that prompts a user for 2 numbers, adds them, and displays the result.


.data
	prompt1: .asciiz "Enter the first number: "
	prompt2: .asciiz "Enter the second number: "
	sum: .asciiz "Result: "
	
.text

.globl main

main:
	li $v0, 4
	la $a0, prompt1
	syscall
	
	li $v0, 5
	syscall
	
	add $s0, $v0, $zero
	li $v0, 4
	la $a0, prompt2
	syscall
	
	li $v0, 5
	syscall
	
	add $s1, $v0, $zero
	add $t0, $s1, $s0
	li $v0, 4
	la $a0, sum
	syscall
	
	add $a0, $t0, $zero
	li $v0, 1
	syscall
	
	li $v0, 10
	syscall
	
	
