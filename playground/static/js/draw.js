// static/js/draw.js
function drawPoints(canvasId, points) {
    var canvas = document.getElementById(canvasId);
    var ctx = canvas.getContext('2d');
    points.forEach(function(point) {
        ctx.fillRect(point[0], point[1], 2, 2);  // Draw a 2x2 rectangle at each point
    });
}