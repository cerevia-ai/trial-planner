import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from PIL import Image

def create_pdf(app_name, control_events, experimental_events, sample_size, effect_size,
               fig_power=None, fig_events=None, logo_bytes=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y_pos = height - inch

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y_pos, app_name)
    y_pos -= 0.5 * inch

    # Optional logo
    if logo_bytes:
        try:
            logo_img = Image.open(io.BytesIO(logo_bytes))
            logo_img = ImageReader(logo_img)
            c.drawImage(logo_img, width - 2.5 * inch, height - 1.5 * inch,
                        width=1.5 * inch, preserveAspectRatio=True, mask='auto')
        except Exception:
            c.setFont("Helvetica", 8)
            c.drawString(1 * inch, y_pos, "⚠️ Invalid logo file – could not display")
            y_pos -= 0.3 * inch

    # Trial details
    c.setFont("Helvetica", 12)
    for line in [
        f"Sample size per group: {sample_size:,}",
        f"Control events: {control_events:.1f}",
        f"Experimental events: {experimental_events:.1f}",
        f"Effect size (Cohen's h): {effect_size:.4f}"
    ]:
        c.drawString(1 * inch, y_pos, line)
        y_pos -= 0.25 * inch

    y_pos -= 0.25 * inch  # extra spacing

    # Helper to add matplotlib figure using PIL
    def add_figure(fig, y_pos, max_width=6*inch, max_height=3*inch):
        if fig is None:
            return y_pos
        buf_img = io.BytesIO()
        fig.savefig(buf_img, format='png', bbox_inches='tight')
        buf_img.seek(0)
        pil_img = Image.open(buf_img)
        img = ImageReader(pil_img)

        # Start new page if not enough space
        if y_pos - max_height < 0.5 * inch:
            c.showPage()
            y_pos = height - inch

        c.drawImage(img, 1*inch, y_pos - max_height, width=max_width,
                    height=max_height, preserveAspectRatio=True, mask='auto')
        pil_img.close()
        buf_img.close()
        return y_pos - max_height - 0.25 * inch

    y_pos = add_figure(fig_power, y_pos)
    y_pos = add_figure(fig_events, y_pos)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(1 * inch, 0.5 * inch, "Generated with Cerevia Clinical Trial Planner")

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes