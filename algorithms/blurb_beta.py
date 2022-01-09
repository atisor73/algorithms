import bokeh.io
import panel as pn

# font-families worth trying: "Open Sans", "Computer Modern"
blurb_style = {'color':'#444444', 'font-size': '14px', "font-family": 'Palatino'}
frac_style = {'color':'#444444', 'font-size': '13px', "font-family": 'Palatino'}

# ****************************************************************************
width_title, width_content = 80, 515
height_story = 90
height_params = 50
height_support = 30
height_pdf = 45
height_cdf = 45
height_moments = 85
height_usage = 60
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
    Say you wait for two multistep Poisson processes to arive. The individual steps of each process happen at the same rate, but the first multistep process requires $\alpha$ steps and the second requires $\beta$ steps. The fraction of the total waiting time taken by the first process is Beta distributed.
    <br><br>
    """, style=blurb_style, width=width_content, height=height_story, margin=0)

m_params_ = pn.pane.LaTeX(r"""
    $\alpha$: # steps in first multi-step process, $(0, \infty)$ <br>
    $\beta$: # steps in second multi-step process, $(0, \infty$) <br><br>
    """, style=blurb_style, width=width_content, height=height_params, margin=0)

m_support_ = pn.pane.LaTeX("""
    on the interval $[0, 1]$ <br><br>
    """, style=blurb_style, width=width_content, height=height_support, margin=0)

m_pdf_ = pn.pane.LaTeX(r"""
    $f(\theta; \alpha, \beta) = \dfrac{\theta^{}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$, &nbsp; where &nbsp; $
    B(\alpha, \beta) = \dfrac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha+\beta)} $<br><br>
    """, style=frac_style, width=width_content, height=height_pdf, margin=0)

m_cdf_ = pn.pane.LaTeX(r"""
    $F(\theta; \alpha, \beta) = I_x (\alpha, \beta) = \dfrac{B(\theta;\alpha, \beta)}{B(\alpha, \beta)}$
    """,style=frac_style, width=width_content, height=height_cdf, margin=0)

m_moments_ = pn.pane.LaTeX(r"""
    $\mathrm{mean}$: $\dfrac{\alpha}{\alpha + \beta}$  <br>
    $\mathrm{variance}$: $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ <br><br>
    """, style=frac_style, width=width_content, height=height_moments, margin=0)

m_usage_ = pn.pane.LaTeX("""
    SciPy: &nbsp;  <code>scipy.stats.beta(alpha, beta)</code> <br>
    Stan: &nbsp;&nbsp;&nbsp; <code>beta(alpha, beta)</code> <br>
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
