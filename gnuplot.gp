reset
#set autoscale
set key left Right nobox samplen 7 lmargin at -2.2,0.95 font ",20"
#set key left nobox samplen 10
#set key outside horizontal bottom center
set title "A*B = C, where A, B, C dimensions are [nxn]" font ",20"
set xlabel "n" font ",20"
set ylabel "time (s)" font ",20"
set tics font ",20"
plot "file2graph.dat" using 2:1 title 'ddot' with lines lw 2, \
     "file2graph.dat" using 3:1 title 'daxpy' with lines lw 2, \
     "file2graph.dat" using 4:1 title 'mmult Kernel' with lines lw 2, \
     "file2graph.dat" using 5:1 title 'cublas Ddot' with lines lw 2, \
     "file2graph.dat" using 6:1 title 'cublas Daxpy' with lines lw 2
     
#set terminal svg size 1600,1200 font ",40"
#set output "sample.svg"
#set terminal jpg color enhanced "Helvetica" 20
set output "output.jpg"
replot
set term x11
