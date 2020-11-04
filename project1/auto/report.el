(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("IEEEtran" "10pt" "conference" "compsocconf")))
   (TeX-run-style-hooks
    "latex2e"
    "IEEEtran"
    "IEEEtran10"
    "hyperref"
    "graphicx")
   (LaTeX-add-labels
    "sec:structure-paper"
    "sec:tips-writing"
    "fig:denoise-fourier"
    "fig:denoise-wavelet"
    "tab:fourier-wavelet"
    "sec:tips-software"
    "sec:latex-primer"
    "eq:linear"))
 :latex)

