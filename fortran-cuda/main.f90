program main 

implicit none 

!define the floating point kind to be single precision
integer, parameter :: fp_kind = kind(0.0)

!define length of the array
integer, parameter :: N=8

complex(fp_kind), dimension(N) :: c, c2
integer :: i

! Initialize array c, compute c2=c*c
do i = 1, N
 c(i) = cmplx(i,2*i)
 c2(i)= c(i)*c(i)
end do

! Print results from Fortran
print *, "Results from Fortran"
do i = 1, N
 print *,i, c(i),c2(i)
end do

! Put 
c2=cmplx(0.,0.)

! Do the same computation with CUDA.
! Fortran -> C -> CUDA ->C ->Fortran
call cudafunction(c,c2,N)

!Results from CUDA
print *, "Results from CUDA"
do i = 1, N
 print *,i, c(i),c2(i)
end do

end program main
