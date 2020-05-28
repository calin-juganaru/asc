function grafice()

  x             = [   400      500      600      700      800      900     1000     1100     1200  ];
  y_neopt       = [0.803924 1.482355 2.571910 4.023327 6.062050 8.558664 11.58065 15.72315 20.00000];
  y_opt_m       = [0.193202 0.291045 0.497348 0.779964 1.156880 1.642699 2.237875 2.972140 3.586991];
  y_opt_f       = [0.175619 0.199846 0.368377 0.646715 1.110594 1.383689 1.930265 2.515020 3.361452];
  y_blas        = [0.055186 0.079524 0.098375 0.148345 0.249132 0.300713 0.405640 0.549388 0.693424];
  y_opt_f_extra = [0.149564 0.163531 0.491946 0.519238 0.966405 1.243190 1.734722 2.329752 3.203181];

  width = 2.75;
  plot(x, y_neopt,        'LineWidth', width,
       x, y_opt_m,        'LineWidth', width,
       x, y_opt_f,        'LineWidth', width,
       x, y_blas,         'LineWidth', width,
       x, y_opt_f_extra,  'LineWidth', width);

end