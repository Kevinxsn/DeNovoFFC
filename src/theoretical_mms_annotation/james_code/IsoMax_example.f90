
implicit none

real*8 :: IsoMaxPercentCutoff

integer :: ierr, counter, i, j, num_iso, iso
real*8 :: parent_mass, mass, intens

integer, dimension(:), allocatable :: AveragineIsoMaxArr, AveragineIsoMinArr
real*8, dimension(:), allocatable :: AveragineMassArr

integer :: arr_ind, IsoMax
real*8 :: FragmentMass



!parameters which must be set in advance
parent_mass=
IsoMaxPercentCutoff=5 !the cutoff used to select isomax, as a percentage of the strongest isotope --- I default to a value of 5


!read the isotopic envelope from the file before generating any fragments

open(unit=2,file='Averagine_Envelopes')
counter=0
!count the number of masses in the file until I find the entry immediately above parent_mass
do while(.true.)
  read(2,*,iostat=ierr) mass, num_iso
  if(is_iostat_end(ierr)) then
    write(*,*) 'ERROR: reached end of averagine envelope file before exceeding parent mass.'
    STOP
  end if
  do j=1, num_iso
    read(2,*)
  end do
  read(2,*)
  counter=counter+1
  if(mass.gt.parent_mass) exit
end do
!allocate the arrays --- note that we could also use IsoMin in an equivalent way to IsoMax, although this only matters for peptides with mass >~7000
allocate(AveragineMassArr(counter),AveragineIsoMaxArr(counter),AveragineIsoMinArr(counter))
AveragineMassArr(:)=0 !list of all masses in the file
AveragineIsoMaxArr(:)=-1 !lowest isotope above intensity cutoff
AveragineIsoMinArr(:)=-1 !highest isotope above intensity cutoff
rewind(2)
do i=1, counter
  read(2,*) AveragineMassArr(i), num_iso
  do j=1, num_iso
    read(2,*) iso, mass, intens
    if((AveragineIsoMinArr(i).eq.-1).and.(intens.ge.IsoMaxPercentCutoff)) then !find lower edge of isotopic envelope
      AveragineIsoMinArr(i)=iso
    end if
    if((AveragineIsoMinArr(i).ne.-1).and.(AveragineIsoMaxArr(i).eq.-1).and.(intens.lt.IsoMaxPercentCutoff)) then !find higher edge of envelope
      AveragineIsoMaxArr(i)=iso-1
    end if
  end do
  if(AveragineIsoMinArr(i).eq.-1) then
    write(*,*) 'ERROR: failed to assign IsoMin when reading averagine envelope file'
    STOP
  end if
  if(AveragineIsoMaxArr(i).eq.-1) then
    AveragineIsoMaxArr(i)=num_iso-1
  end if
end do
close(2)


!now IsoMax is known for all masses in the file up to the entry immediately above the parent mass



!NOW ENTER INTO FRAGMENT GENERATION PART; for each fragment use the following to get IsoMax

FragmentMass= !whatever the frament mass is here

call get_nearest_value(FragmentMass,AveragineMassArr,arr_ind)
IsoMax=AveragineIsoMaxArr(arr_ind)


!uses the subroutine:

contains

subroutine get_nearest_value(val,arr_to_compare,frag_ind)

!gets the index frag_ind of the nearest value to the input val in the array arr_to_compare

implicit none

real*8 :: val
real*8, dimension(:) :: arr_to_compare
integer :: dimen, frag_ind
integer :: ind_current, ind_upper, ind_lower

ind_upper=size(arr_to_compare)
ind_lower=1

do while(.true.)
  ind_current=(ind_upper+ind_lower)/2 !integer division
  if(val.gt.arr_to_compare(ind_current)) then
    ind_lower=ind_current
  else
    ind_upper=ind_current
  end if
  if(ind_upper-ind_lower.le.1) then
    if(abs(val-arr_to_compare(ind_upper)).lt.abs(val-arr_to_compare(ind_lower))) then
      frag_ind=ind_upper
    else
      frag_ind=ind_lower
    end if
    exit
  end if
end do


end subroutine get_nearest_value

