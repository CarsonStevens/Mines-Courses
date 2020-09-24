.data
	a: .word 1, 5, 12, -1, 15, 18, 33, 7, 0, 222
	prompt1: .asciiz "Enter an index from 1 to 8: "
	prompt2: .asciiz "Value at index i before modification: "
	sum: .asciiz "Value at index i after modification: "
	newline: .asciiz "\n"
.text
.globl main
main:
	# Your code
	la $t3, a		#list a address is in $t3
	 
	   
	li $v0, 4		#prints the prompt to ask for index
	la $a0, prompt1
	syscall
	
	li $v0, 5
	syscall
	add $s1, $v0, $zero	#$s1 has value of i
	sll $s1, $s1, 2		#$s1 now has the index*offset(4)
	add $t3, $s1, $t3	#$t3 now has the base address plus the offset
	lw $s1, 0($t3)		#s1 now has the value at the offset
	

	li $v0, 4		#prompts users of the index value before mod
	la $a0, prompt2
	syscall
	
	add $a0, $s1, $zero	#$a0 now has the value at the array[i]
	li $v0, 1		#prints the value of array[i]
	syscall
	
	li $v0, 4      		#prints a newline
	la $a0, newline  
	syscall
	
	lw $t1, 4($t3)		#loads a[i+1] into $t1
	lw $t2, -4($t3)		#loads a[i-1] into $t2
	add $t4, $t2, $t1	#$t4 = $t2 + $t1
	
	sw $t4, 0($t3)		#stores the result back into the array at index i
	
	li $v0, 4		#prints after mod prompt
	la $a0, sum
	syscall
	
	add $a0, $t4, $zero	#stores mod into $a0 and prints it
	li $v0, 1
	syscall
	 
	# Exit  
	li $v0, 10
	syscall
	 
