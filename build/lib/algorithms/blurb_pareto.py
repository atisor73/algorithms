import bokeh.io
import panel as pn

# font-families worth trying: "Open Sans", "Computer Modern"
blurb_style = {'color':'#444444', 'font-size': '14px', "font-family": 'Palatino'}
frac_style = {'color':'#444444', 'font-size': '13px', "font-family": 'Palatino'}

# ****************************************************************************
width_title, width_content = 80, 515
height_story = 45
height_params = 50
height_support = 30
height_pdf = 45
height_cdf = 45
height_moments = 145
height_usage = 50
height_comments = 0


# LHS TITLES, RIGHT ALIGN
m_story = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **Story**: """, style=blurb_style, width=width_title, height=height_story, height_policy="fixed", margin=0)
m_params = pn.pane.Markdown("""**Parameters**: """,
    style=blurb_style, width=width_title, height=height_params, height_policy="fixed", margin=0)
m_support = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;
    **Support**: """, style=blurb_style, width=width_title, height=height_support, height_policy="fixed", margin=0)
m_pdf = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **PDF**: """, style=blurb_style, width=width_title, height=height_pdf, height_policy="fixed", margin=0)
m_cdf = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **CDF**: """, style=blurb_style, width=width_title, height=height_cdf, height_policy="fixed", margin=0)
m_moments = pn.pane.Markdown("""&nbsp;&nbsp;
    **Moments**: """, style=blurb_style, width=width_title, height=height_moments, height_policy="fixed", margin=0)
m_usage = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **Usage**: """,style=blurb_style, width=width_title, height=height_usage, height_policy="fixed", margin=0)
m_comments = pn.pane.Markdown("""
    **Comments**: """,style=blurb_style, width=width_title, height=height_comments, height_policy="fixed", margin=0)



# RHS CONTENT
m_story_ = pn.pane.LaTeX(r"""
    A distribution where the tail of the PDF or PMF follows a power law $f(y) \sim y^{-\alpha-1}$.
    Such distributions often arise in physical scenarios.<br><br>
    """, style=blurb_style, width=width_content, height=height_story, margin=0)

m_params_ = pn.pane.LaTeX(r"""
    $\alpha$: sets the power in the power law, $(0, \infty)$ <br>
    $y_{\mathrm{min}}$: lower cutoff ensures normalization, $[0, \infty$) <br><br>
    """, style=blurb_style, width=width_content, height=height_params, margin=0)

m_support_ = pn.pane.LaTeX("""
    set of real numbers $\mathbb{R} \geq y_{\mathrm{min}}$  <br><br>
    """, style=blurb_style, width=width_content, height=height_support, margin=0)

m_pdf_ = pn.pane.LaTeX(r"""
 $f(y; y_{\mathrm{min}}, \alpha) = \dfrac{\alpha}{y} \left(\dfrac{y_{\mathrm{min}}}{y} \right)^\alpha$ <br><br>
    """, style=frac_style, width=width_content, height=height_pdf, margin=0)

m_cdf_ = pn.pane.LaTeX(r"""
    $F(y; y_{\mathrm{min}}, \alpha)= 1-\left( \dfrac{y_{\mathrm{min}}}{y} \right)^{\alpha}$
    """, style=blurb_style, width=width_content, height=height_cdf, margin=0)

m_moments_ = pn.pane.LaTeX(r"""
    $\mathrm{mean}$: $\begin{cases}
        \infty & \alpha \leq 1 \\
        \dfrac{\alpha y_{\mathrm{min}}}{\alpha-1} & \alpha > 1
        \end{cases}
        $  <br><br>
    $\mathrm{variance}$: $\begin{cases}
            \infty & \alpha \leq 2 \\
            \dfrac{\alpha y_{\mathrm{min}}^2}{(\alpha-1)^2(\alpha -2)} & \alpha > 2
        \end{cases}
        $ <br><br>
    """, style=frac_style, width=width_content, height=height_moments, margin=0)

m_usage_ = pn.pane.LaTeX("""
    SciPy: &nbsp;  <code>scipy.stats.pareto(alpha, scale=y_min)</code> <br>
    Stan: &nbsp;&nbsp;&nbsp; <code>pareto(y_min, alpha)</code> <br>
    """, style=blurb_style, width=width_content, height=height_usage, margin=0)

m_comments_ = pn.pane.LaTeX("""
    <br><br>
    """, style=blurb_style, width=width_content, height=height_comments, margin=0)

m_all = pn.Column(
    pn.Row(m_story, m_story_),
    pn.Row(m_params, m_params_),
    pn.Row(m_support, m_support_),
    pn.Row(m_pdf, m_pdf_),
    pn.Row(m_cdf, m_cdf_),
    pn.Row(m_moments, m_moments_),
    pn.Row(m_usage, m_usage_),
    pn.Row(m_comments, m_comments_),
)

desc = pn.Row(pn.Spacer(width=15), pn.Column(pn.Spacer(height=25), m_all))
