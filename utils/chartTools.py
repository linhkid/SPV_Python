import os
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts import LineChart
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def saveChartAsPDF(chart, filename, width=640, height=480):
    drawing = Drawing(width, height)
    drawing.add(chart)
    renderPDF.drawToFile(drawing, filename)

def saveChartAsPDF(chart, filename, width, height):
    drawing = Drawing(width, height)
    drawing.add(chart)
    renderPDF.drawToFile(drawing, filename)


