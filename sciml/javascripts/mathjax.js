// MathJax configuration for SciML documentation
// Workshop-inspired scientific equation rendering

window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
    // Scientific computing specific macros
    macros: {
      // Vector and matrix notation
      "vec": ["\\mathbf{#1}", 1],
      "mat": ["\\mathbf{#1}", 1],
      "norm": ["\\left\\|#1\\right\\|", 1],
      "abs": ["\\left|#1\\right|", 1],

      // Differential operators
      "grad": "\\nabla",
      "div": "\\nabla \\cdot",
      "curl": "\\nabla \\times",
      "laplacian": "\\nabla^2",

      // Partial derivatives
      "pd": ["\\frac{\\partial #1}{\\partial #2}", 2],
      "pdd": ["\\frac{\\partial^2 #1}{\\partial #2^2}", 2],

      // Neural network notation
      "NN": "\\mathcal{N}",
      "loss": "\\mathcal{L}",
      "params": "\\theta",

      // Physics notation
      "hamiltonian": "\\hat{H}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Support for Material theme's instant loading feature
// This ensures MathJax re-renders when navigating between pages
document.addEventListener('DOMContentLoaded', function() {
  if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
      if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        MathJax.startup.output.clearCache()
        MathJax.typesetClear()
        MathJax.texReset()
        MathJax.typesetPromise()
      }
    });
  }
});
