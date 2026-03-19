function(cluster) {
    var markers = cluster.getAllChildMarkers();
    var counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0};
    var dominantColor = 'darkgreen';
    var total_markers = 0;

    markers.forEach(function(marker) {
        total_markers++;
        // Check for DivIcon (TP, FP, FN)
        if (marker.options.icon && marker.options.icon.options.html) {
            var outcome = marker.options.icon.options.html.match(/data-outcome="([^"]*)"/);
            if (outcome) {
                counts[outcome[1]]++;
            }
        }
        // Check for Circle Marker (TN - simple color/radius)
        else if (marker.options.color === 'gray' && marker.options.radius === 3) {
            counts['TN']++;
        }
    });

    var total_non_tn = counts.TP + counts.FP + counts.FN;
    var total_all = total_markers;

    // 1. TN Dominance (if > 75% of points are TN)
    if (total_all > 0 && counts.TN / total_all > 0.75) {
        dominantColor = 'gray';
    }
    // 2. Non-TN Dominance Logic
    else if (total_non_tn > 0) {
        var tolerance = 0.10;
        // FN Dominance
        if (counts.FN > counts.TP && counts.FN > counts.FP) {
            dominantColor = 'orange';
        }
        // FP Dominance
        else if (counts.FP > counts.TP && counts.FP > counts.FN) {
            dominantColor = 'red';
        }
        // TP/FP Mixed Zone
        else if (Math.abs(counts.TP - counts.FP) / total_non_tn < tolerance) {
            dominantColor = 'orange';
        }
        // TP Dominance
        else {
            dominantColor = 'green';
        }
    } else {
        dominantColor = 'gray';
    }

    return L.divIcon({
        html: '<div style="background-color:' + dominantColor + '; color: white; border-radius: 50%; width: 40px; height: 40px; line-height: 40px; text-align: center; font-weight: bold;">' + cluster.getChildCount() + '</div>',
        className: 'marker-cluster',
        iconSize: new L.Point(40, 40)
    });
}
